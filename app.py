

from langchain_community.document_loaders import UnstructuredPDFLoader  # For loading PDF documents
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting documents into chunks
from langchain_chroma import Chroma  # For vector storage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # For embeddings and language models
from langchain_core.prompts import ChatPromptTemplate  # For LLM prompts
from langchain_core.runnables import RunnablePassthrough  # For RAG chain connections

# Load the PDF as a LangChain document loader
loader = UnstructuredPDFLoader(file_path="/data/mg-zs-warning-messages.pdf")
car_docs = loader.load()

# Initialize RecursiveCharacterTextSplitter to make chunks of PDF text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Split PDF into smaller chunks
splits = text_splitter.split_documents(car_docs)

# Initialize Chroma vectorstore with documents as splits and using OpenAIEmbeddings
openai_api_key = "YOUR_API_KEY"
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(openai_api_key=openai_api_key))

# Setup vectorstore as retriever
retriever = vectorstore.as_retriever()

# Define RAG prompt
prompt = ChatPromptTemplate.from_template(
    "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \nContext: {context} \nAnswer:"
)

# Initialize chat-based LLM with 0 temperature and using gpt-4o-mini
model = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o-mini", temperature=0)

# Setup the RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
)

# Initialize query
query = "The Gasoline Particular Filter Full warning has appeared. What does this mean and what should I do about it?"

# Invoke the query
answer = rag_chain.invoke(query).content

# Print the answer
print(answer)
