import re
from typing import Optional, List, Dict, Any, Tuple


from ..core.agent import Agent
from ..core.llm import HelloAgentsLLM
from ..core.config import Config
from ..core.message import Message



REACT_PROMPT_TEMPLATE = """
你是一个具备推理和行动功能的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具如下：
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤：

- Thought:分析问题，确定需要什么信息，制定研究策略。
- Action:判断是否需要调用外部工具，还是已经有足够的信息得到问题的结果

## 重要提醒
1. 每次回应必须包含Thought和Action两部分；
2. 工具调用的格式必须遵循：工具名[参数]；
3. 只有当你确信有足够信息回答问题时，才使用Finish
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数

## 输出格式
- 如果需要调用指定工具：
Thought:...
Action:{{tool_name}}[{{tool_input}}]
- 如果能确定最终的答案：
Thought:...
Action:Finish[研究结论]

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动
"""


class ReActAgent:
    """
    ReAct(Reasoning and Acting) Agent
    核心：思考-行动-观察
    结合推理和行动的智能体，能够：
    1. 分析问题并制定计划
    2. 调用外部工具获取信息
    3. 基于观察结果进行推理
    4. 迭代执行直到得出最终答案
    """
    def __init__(self,
        llm_client:HelloAgentsLLM,
        tool_executor: ToolExecutor,
        max_steps: int=5
    ):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def _parse_output(self, text:str) -> Tuple[Optional[str], Optional[str]]:
        """解析LLM的输出，提取Thought和Action"""
        thought_match = re.search(r"Thought:(.*)", text)
        action_match = re.search(r"Action:(.*)", text)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str) -> Tuple:
        """解析Action字符串，提取工具名称和输入"""
        _match = re.match(r"(\w+)\[(.*)\]", action_text)
        if _match:
            return _match.group(1), _match.group(2)

        return None, None

    def _parse_action_input(self, action_text: str) -> str:
        """解析行动输入"""
        match = re.match(r"\w+\[(.*)\]", action_text)
        return match.group(1) if match else ""


    def run(self, question:str):
        """运行ReAct智能体来回答一个问题"""
        self.history = []     # 每次运行都重置历史记录
        current_step = 0

        print(f"\n🤖 开始处理问题: {question}")

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = '\n'.join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )
            # 2. 调用LLM进行思考
            messages = [{'role': 'user', 'content': prompt}]
            response_text = self.llm_client.think(messages)

            if not response_text:
                print("错误：LLM未正常响应。")
                break

            thought, action = self._parse_output(response_text)
            if thought:
                print(f"思考：{thought}\n行动：{action}")
            if not action:
                print(f"未能解析出有效Action,流程终止。:{response_text}")
                break

            # 4. 执行Action
            if action.startswith("Finish"):
                # 提取出最终答案并结束
                final_answer = self._parse_action_input(action)
                print(f"🎉 最终答案: {final_answer}")
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            if not tool_name or tool_input is None:
                print("无效Action")
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")

            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"错误:未找到名为 '{tool_name}' 的工具。"
            else:
                observation = tool_function(tool_input) # 调用真实工具

            print(f"👀 观察: {observation}")

            # 将本轮的Action和Observation添加到历史记录中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        # 循环结束
        print("已达到最大步数，流程终止。")
        return None