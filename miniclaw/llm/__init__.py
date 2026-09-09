"""模型协议层：消息域对象与（后续切片的）模型客户端适配器。"""

from miniclaw.llm.messages import Message, ToolCall, Usage

__all__ = ["Message", "ToolCall", "Usage"]
