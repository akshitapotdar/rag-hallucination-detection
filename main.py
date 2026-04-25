from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.language_models.llms import LLM
from typing import Optional, List, Any
import numpy as np


# Load data
loader = TextLoader("data.txt")
docs = loader.load()

# Split text
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Vector DB
db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever()


# ========== LLM ==========
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


class FlanT5LLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "flan-t5"

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Any = None, **kwargs: Any) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=80)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


llm = FlanT5LLM()


print("\n===== OUTPUT =====")

questions = [
    "What is AI?",
    "What is machine learning?",
    "Who invented AI?",
    "What is neural network?"
]

results = []

for q in questions:
    retrieved_docs = retriever.invoke(q)
    context = " ".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""Context: {context}

Question: {q}

Give a short, direct answer based only on the context."""

    answer = llm.invoke(prompt).strip()

    results.append({
        "question": q,
        "answer": answer,
        "context": context
    })

    print("\nQ:", q)
    print("A:", answer)



def overlap_score(answer, context):
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    if not answer_words:
        return 0
    return len(answer_words.intersection(context_words)) / len(answer_words)



def llm_judge(answer, context, question):
    prompt = f"""Context: {context}

Question: {question}
Answer: {answer}

Is the answer fully supported by the context? Reply only YES or NO."""
    return llm.invoke(prompt).strip().upper()


def semantic_similarity(answer, context):
    if not answer.strip():
        return 0.0
    answer_vec = np.array(embeddings.embed_query(answer))
    context_vec = np.array(embeddings.embed_query(context))

    cosine = np.dot(answer_vec, context_vec) / (
        np.linalg.norm(answer_vec) * np.linalg.norm(context_vec)
    )
    return float(cosine)



def specificity_score(answer):
    word_count = len(answer.split())
    if word_count <= 2:
        return "LOW"
    elif word_count <= 4:
        return "MEDIUM"
    else:
        return "HIGH"



def hallucination_flag(overlap, similarity, judge, specificity):
   
    if specificity == "LOW":
        return "LOW_INFO"
    
    if "NO" in judge or similarity < 0.4:
        return "HALLUCINATION"
    
    if overlap >= 0.5 and similarity >= 0.6 and "YES" in judge:
        return "GROUNDED"
    
    return "UNCERTAIN"


# ========== RUN ANALYSIS ==========
print("\n\n===== HALLUCINATION ANALYSIS =====")

for r in results:
    overlap = overlap_score(r["answer"], r["context"])
    similarity = semantic_similarity(r["answer"], r["context"])
    judge = llm_judge(r["answer"], r["context"], r["question"])
    specificity = specificity_score(r["answer"])
    flag = hallucination_flag(overlap, similarity, judge, specificity)

    print("\n----------------------")
    print("Q:           ", r["question"])
    print("A:           ", r["answer"])
    print("Overlap:     ", round(overlap, 2))
    print("Similarity:  ", round(similarity, 2))
    print("LLM Judge:   ", judge)
    print("Specificity: ", specificity)
    print("Verdict:     ", flag)