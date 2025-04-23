# 使用qwen
from langchain_community.llms import Tongyi
from langchain.schema import HumanMessage, SystemMessage, AIMessage, FunctionMessage
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_community.chat_models import ChatTongyi
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.chains import LLMChain
# from getpass import getpass
import os

DASHSCOPE_API_KEY = 'sk-0586aa6d1c104130b57fa50af0c869b8'
os.environ['DASHSCOPE_API_KEY'] = DASHSCOPE_API_KEY

# 使用llms模型
llm = Tongyi()
# res = llm("我有一些问题要问你")
# print(res)

# 使用chatModel模型
chat_model = ChatTongyi()
# rres = chat_model.predict("你是什么模型？")
# # 上面默认转化为HumanMessage
# print(rres)

# Message方式
# messages = [
#     SystemMessage("我是一个天气预报机器人"),
#     HumanMessage("你好，你能做什么？")
# ]
# res3 = chat_model.predict_messages(messages)
# print(res3)

# prompt工程
system_message_prompt = SystemMessagePromptTemplate.from_template("我是一个天气预报机器人")
human_message_prompt = HumanMessagePromptTemplate.from_template("我想查{area}的天气")
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
#
# # 用户输入 area 的值
# area = input("请输入您想查询的地区：")
# prompt = chat_prompt.format_prompt(area=area)
# res4 = llm.invoke(prompt)
#
# # 自定义输出解析器
# class WeatherOutputParser(BaseOutputParser):
#     def parse(self, text: str):
#         import re
#
#         # 定义正则表达式模式
#         patterns = {
#             "temperature": r"当前温度约为(\d+)°C",
#             "high_temperature": r"最高温度约(\d+)°C",
#             "low_temperature": r"最低温度约(\d+)°C",
#             "condition": r"天气状况：([\u4e00-\u9fa5]+)",
#             "humidity": r"相对湿度在(\d+)%左右",
#             "wind": r"风速：([\u4e00-\u9fa50-9-]+)",
#         }
#
#         # 提取匹配结果
#         result = {}
#         for key, pattern in patterns.items():
#             match = re.search(pattern, text)
#             if match:
#                 result[key] = match.group(1)
#             else:
#                 result[key] = None  # 如果未找到匹配项，设置为 None
#
#         return result
#
# parser = WeatherOutputParser()
#
# parse_res = parser.parse(res4)
# for key, value in parse_res.items():
#     print(f"{key.capitalize().replace('_', ' ')}: {value}")

# LLMChain
chain=LLMChain(
    llm=Tongyi(),
    prompt=chat_prompt
)
res6 = chain.run("上海")
print(res6)

