import asyncio

from langchain_community.llms import Tongyi
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser

import os

DASHSCOPE_API_KEY = 'sk-0586aa6d1c104130b57fa50af0c869b8'
os.environ['DASHSCOPE_API_KEY'] = DASHSCOPE_API_KEY

# LCEL基本示例 BasePromptTemplate | model | BaseOutputParser
# 这三个接口的实现类都实现了invoke
# model的invoke参数可以是PromptValue，返回是BaseMessage
# BaseOutputParser的invoke的参数可以是BaseMessage，返回是str
# invoke在这里是在输入上同步的调用
chat_prompt = ChatPromptTemplate.from_template("给我推荐歌手{singer}的几首代表作")
model = Tongyi()
output_parser = StrOutputParser()

chain = chat_prompt | model | output_parser

# res1 = chain.invoke({"singer": "陈百强"})
# print(res1)

# 同步方法stream（流式）、invoke、batch（输入列表）
# 实现Runnable接口的组件有Prompt、ChaModel、LLM、OutputParser、Retriever、Tool

# stream
# for s in chain.stream({"singer": "陈百强"}):
#     print(s, end="", flush=True)

# batch
# res2 = chain.batch([{"singer": "张国荣"}, {"singer": "王菲"}])
# print(res2)

# async stream
# async def asyncOutput():
#     async for s1 in chain.astream({"singer": "陈百强"}):
#         print(s1, end="", flush=True)
#
# asyncio.run(asyncOutput())

# 并发执行
async def stream_output(singer, delay=0):
    async for chunk in chain.astream({"singer": singer}):
        print(f"[{singer}] {chunk}", end="", flush=True)
        await asyncio.sleep(delay)  # 模拟延迟，便于观察交错效果

async def main():
    # 使用 asyncio.gather 并发运行
    await asyncio.gather(
        stream_output("陈百强", delay=0.1),
        stream_output("王菲", delay=0.1)
    )

asyncio.run(main())