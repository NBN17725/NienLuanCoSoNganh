import json
import warnings
import chromadb
import torch
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoTokenizer, logging

# 1. Tắt cảnh báo
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 2. Load các mô hình AI
print("Đang nạp mô hình GLiClass...")
model_name = "knowledgator/gliclass-instruct-large-v1.0"
model = GLiClassModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipeline = ZeroShotClassificationPipeline(
    model, tokenizer, classification_type="multi-label", device=device
)

print("Đang nạp mô hình BAAI/bge-m3 & Reranker...")
embedding_model = SentenceTransformer("BAAI/bge-m3")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

# 3. Khởi tạo ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="knowledge_base")


# 4. Hàm chia nhỏ văn bản (Chunking)
def chunk_text(text, tokenizer, chunk_size=150, overlap=30):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = []
    current_tokens = 0

    for paragraph in paragraphs:
        paragraph_tokens = tokenizer.encode(paragraph, add_special_tokens=False)

        if len(paragraph_tokens) > chunk_size:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_tokens = 0
            start = 0
            while start < len(paragraph_tokens):
                end = start + chunk_size
                chunk_tokens = paragraph_tokens[start:end]
                chunk = tokenizer.decode(
                    chunk_tokens, skip_special_tokens=True
                )
                if chunk.strip():
                    chunks.append(chunk)
                start += chunk_size - overlap
            continue

        if current_tokens + len(paragraph_tokens) <= chunk_size:
            current.append(paragraph)
            current_tokens += len(paragraph_tokens)
        else:
            if current:
                chunks.append("\n\n".join(current))
            current = [paragraph]
            current_tokens = len(paragraph_tokens)

    if current:
        chunks.append("\n\n".join(current))
    return chunks


# 5. Kiểm tra và Nạp KB vào ChromaDB nếu chưa có
if collection.count() == 0:
    KB_PATH = r"C:\Users\PC\Desktop\chayNienLuan\Knowledge_base\Knowledge_Base.json"
    print("Đang đọc Knowledge Base...")
    with open(KB_PATH, "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)

    ids, documents, metadatas = [], [], []

    for item in knowledge_base:
        doc_id = item.get("id")
        context = item.get("context", "")
        if not context.strip():
            continue

        chunks = chunk_text(context, tokenizer, chunk_size=150, overlap=30)
        domain = item.get("domain", "")
        topic = item.get("topic", "")
        title = item.get("title", "")

        for chunk_index, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{chunk_index}"
            ids.append(chunk_id)

            document_for_embedding = (
                f"Chủ đề: {topic}. Tiêu đề: {title}.\nNội dung: {chunk}"
            )
            documents.append(document_for_embedding)

            metadatas.append({
                "kb_id": doc_id,
                "chunk_id": chunk_index,
                "domain": domain,
                "topic": topic,
                "title": title,
                "source": item.get("source", ""),
                "url": item.get("url", ""),
                "raw_chunk": chunk,
            })

    print(f"Đang tạo embedding cho {len(documents)} chunks...")
    embeddings = embedding_model.encode(
        documents, show_progress_bar=True
    ).tolist()

    BATCH_SIZE = 2000
    for start in range(0, len(ids), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(ids))
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )
    print("Đã lưu thành công dữ liệu vào ChromaDB.")
else:
    print(f"ChromaDB đã có sẵn {collection.count()} chunks. Bỏ qua bước nạp KB.")


# 6. Hàm xử lý kiểm tra Ảo giác
RERANK_THRESHOLD = 0.1

def check_hallucination(answer_text: str):
    full_answer = answer_text.strip()

    # Cắt Answer thành các chunk nhỏ
    answer_chunks = chunk_text(full_answer, tokenizer, chunk_size=80, overlap=15)
    
    best_candidate = None
    max_rerank_score = -999.0

    # Lặp qua các chunks để tìm ngữ cảnh (Context) phù hợp nhất
    for ans_chunk in answer_chunks:
        chunk_embedding = embedding_model.encode(ans_chunk).tolist()
        search_results = collection.query(
            query_embeddings=[chunk_embedding],
            n_results=10,
            include=["documents", "metadatas", "distances"],
        )

        candidate_documents = search_results["documents"][0]
        candidate_metadatas = search_results["metadatas"][0]
        candidate_distances = search_results["distances"][0]

        candidates = []
        for doc, metadata, distance in zip(candidate_documents, candidate_metadatas, candidate_distances):
            candidates.append({"document": doc, "metadata": metadata, "distance": distance})

        # Đối chiếu trực tiếp Tiêu đề
        ans_chunk_lower = ans_chunk.lower()
        for item in candidates:
            title = item["metadata"].get("title", "").strip().lower()
            if title and (title in ans_chunk_lower or any(word in ans_chunk_lower for word in title.split() if len(word) > 3)):
                item["title_matched"] = True
            else:
                item["title_matched"] = False

        # Rerank & Cộng điểm ưu tiên
        pairs = [[ans_chunk, item["document"]] for item in candidates]
        rerank_scores = reranker.predict(pairs)

        for item, score in zip(candidates, rerank_scores):
            item["rerank_score"] = float(score)
            if item["title_matched"]:
                item["rerank_score"] += 2.5

        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        if ranked and ranked[0]["rerank_score"] > max_rerank_score:
            max_rerank_score = ranked[0]["rerank_score"]
            best_candidate = ranked[0]

    # Lọc ngữ cảnh Context
    if best_candidate and max_rerank_score >= RERANK_THRESHOLD:
        context = best_candidate["metadata"].get("raw_chunk", best_candidate["document"])
        ctx_title = best_candidate["metadata"].get("title", "Không xác định")
        ctx_score = round(max_rerank_score, 4)
    else:
        context = "Không tìm thấy thông tin liên quan trong Cơ sở dữ liệu đối soát."
        ctx_title = "Không tìm thấy"
        ctx_score = 0.0

    # Đưa vào GLiClass đánh giá Ảo giác
    input_text = f"Context: {context}\nAnswer: {full_answer}"
    labels = ["hallucinated", "correct"]
    gliclass_res = pipeline(input_text, labels, threshold=0.0)[0]

    scores = {"correct": 0.0, "hallucinated": 0.0}
    for item in gliclass_res:
        label = item["label"]
        if label in scores:
            scores[label] = round(float(item["score"]), 4)

    is_hallucinated = scores["hallucinated"] > scores["correct"]

    return {
        "is_hallucinated": is_hallucinated,
        "context": context,
        "context_info": {
            "title": ctx_title,
            "score": ctx_score
        },
        "scores": scores
    }