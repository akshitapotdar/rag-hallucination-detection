from transformers import pipeline
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline


# Load data
loader = TextLoader("data.txt")
docs = loader.load()

# Split text
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Embeddings
embeddings = HuggingFaceEmbeddings()

# Vector DB
db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever()

# LLM (better than distilgpt2)
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=50
)
llm = HuggingFacePipeline(pipeline=pipe)


# RAG chain
print("\n===== OUTPUT =====")

questions = [
    "What is AI?",
    "What is machine learning?",
    "Who invented AI?",
    "What is neural network?"
]

for q in questions:
    docs = retriever.invoke(q)
    context = " ".join([doc.page_content for doc in docs])

    prompt = f"""
    Context: {context}

    Question: {q}

    Give a short, direct answer based only on the context.
    """

    result = llm.invoke(prompt)
    answer = result if isinstance(result, str) else result["text"]

    print("\nQ:", q)
    print("A:", answer)