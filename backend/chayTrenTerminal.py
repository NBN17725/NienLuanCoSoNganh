import json
import warnings
import chromadb
import torch
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoTokenizer, logging

# 1. Tắt các cảnh báo không cần thiết
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 2. Tải mô hình GLiClass
model_name = "knowledgator/gliclass-instruct-large-v1.0"
model = GLiClassModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipeline = ZeroShotClassificationPipeline(
    model, tokenizer, classification_type="multi-label", device=device
)

# 3. Nạp mô hình Embedding & Reranker
print("Đang nạp mô hình BAAI/bge-m3 & Reranker...")
embedding_model = SentenceTransformer("BAAI/bge-m3")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

# 4. Khởi tạo ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="knowledge_base")


# 5. Hàm chia nhỏ văn bản
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


# 6. Kiểm tra và Nạp KB
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


# 7. Vòng lặp kiểm tra Ảo giác: Cắt Answer -> Đối chiếu Tiêu đề -> Lấy Context -> GLiClass
RERANK_THRESHOLD = 0.1

while True:
    print("\n--------------------------------------------------")
    full_answer = input("Nhập câu trả lời cần kiểm tra: ")
    if not full_answer.strip():
        continue

    # 7.1. Cắt Answer thành các chunk nhỏ (ví dụ 60-100 tokens)
    answer_chunks = chunk_text(full_answer, tokenizer, chunk_size=80, overlap=15)
    print(f"\n[THÔNG BÁO] Đã cắt Answer thành {len(answer_chunks)} chunk(s).")

    for idx, ans_chunk in enumerate(answer_chunks, 1):
        print(f"\n>>> DANG KIỂM TRA CHUNK {idx}/{len(answer_chunks)}: \"{ans_chunk}\"")

        # 7.2. Truy vấn Vector Top 10 ứng với Chunk này
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
        for doc, metadata, distance in zip(
            candidate_documents, candidate_metadatas, candidate_distances
        ):
            candidates.append(
                {"document": doc, "metadata": metadata, "distance": distance}
            )

        # 7.3. ĐỐI CHIẾU TRỰC TIẾP CHUNK VỚI TIÊU ĐỀ (TITLE MATCHING)
        ans_chunk_lower = ans_chunk.lower()
        matched_by_title = []

        for item in candidates:
            title = item["metadata"].get("title", "").strip()
            title_lower = title.lower()

            # Kiểm tra xem Tiêu đề có nằm trong Chunk hoặc ngược lại từ khóa Tiêu đề khớp với Chunk không
            if title_lower and (title_lower in ans_chunk_lower or any(word in ans_chunk_lower for word in title_lower.split() if len(word) > 3)):
                item["title_matched"] = True
            else:
                item["title_matched"] = False

        # 7.4. Đánh giá lại thứ hạng bằng Reranker & Cộng điểm ưu tiên nếu TRÙNG TIÊU ĐỀ
        pairs = [[ans_chunk, item["document"]] for item in candidates]
        rerank_scores = reranker.predict(pairs)

        for item, score in zip(candidates, rerank_scores):
            item["rerank_score"] = float(score)
            if item["title_matched"]:
                item["rerank_score"] += 2.5  # Thưởng điểm cao cho các candidate TRÙNG TIÊU ĐỀ

        # Sắp xếp lại theo điểm Rerank (đã cộng điểm thưởng Tiêu đề)
        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        top_candidate = ranked[0] if ranked else None

        # 7.5. Lọc ngữ cảnh Context
        if top_candidate and top_candidate["rerank_score"] >= RERANK_THRESHOLD:
            context = top_candidate["metadata"].get(
                "raw_chunk", top_candidate["document"]
            )
            has_valid_context = True
        else:
            context = "Không tìm thấy thông tin liên quan trong Cơ sở dữ liệu."
            has_valid_context = False

        print("\n================ KẾT QUẢ ĐỐI CHIẾU TIÊU ĐỀ & CONTEXT ================")
        if has_valid_context and top_candidate:
            print(f"Tiêu đề KB: {top_candidate['metadata'].get('title')}")
            print(f"Trạng thái trùng khớp Tiêu đề: {'Có khớp' if top_candidate['title_matched'] else 'Không khớp trực tiếp'}")
            print(f"Điểm Rerank tổng hợp: {top_candidate['rerank_score']:.4f}")
            print(f"Context: {context}")
        else:
            print("⚠️ CẢNH BÁO: Không tìm thấy Context phù hợp!")
            print(f"Context: {context}")

        # 7.6. Đưa Context và Answer Chunk vào GLiClass kiểm tra Ảo giác
        text = f"Context: {context}\nAnswer: {full_answer}"
        labels = ["hallucinated", "correct"]

        results = pipeline(text, labels, threshold=0.0)[0]

        print("\n================ ĐÁNH GIÁ ẢO GIÁC (GLICLASS) ================")
        for r in results:
            print(f"{r['label']:<15} => {r['score']:.4f}")
        print("====================================================================")