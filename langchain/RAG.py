from langchain_community.llms import Tongyi
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import Document
import os

DASHSCOPE_API_KEY = 'sk-0586aa6d1c104130b57fa50af0c869b8'
os.environ['DASHSCOPE_API_KEY'] = DASHSCOPE_API_KEY

# textLoader加载翁当资源
loader = TextLoader("./tinyLLM.txt")
documents = loader.load()

# 分割文档为更小的块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=50)
documents_text = [doc.page_content for doc in documents]
texts = text_splitter.split_text("\n".join(documents_text))

# 将文本片段包装为文档对象
docs = [Document(page_content=text) for text in texts]

# 嵌入模型生成向量
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# FAISS向量存储和检索
# for_documents接口：1. 遍历texts中的每个文档 2. 使用embeddings将文档转化为向量 3. 存入FAISS向量数据库
vector_store = FAISS.from_documents(docs, embeddings)

# 初始化检索器
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

llm = Tongyi()
# 构建RAG链
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

system_prompt = SystemMessagePromptTemplate.from_template("我是对Tiny LLM这个项目非常熟悉的机器人")
human_prompt = HumanMessagePromptTemplate.from_template("关于Tiny LLM这个项目，我想问一下：{input}")

chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
input_str = chat_prompt.format_prompt(input="这个项目是干什么的？").to_string()

res = rag_chain.invoke({"query": input_str})
# 输出结果
print("AI 回答：")
print(res["result"])

print("\n参考文档片段：")
for doc in res["source_documents"]:
    print(f"- {doc.page_content[:100]}...")