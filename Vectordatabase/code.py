import numpy as np

def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))

def dot_product(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.dot(u, v))

def euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.linalg.norm(u - v))

def demo_metrics():
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([2.0, 4.0, 6.0])   
    w = np.array([3.0, -1.0, 0.0])  

    print(f"Vector u: {u}")
    print(f"Vector v: {v}")
    print(f"Vector w: {w}\n")

    print(f"- Cosine(u, v):    {cosine_similarity(u, v):.4f} (Trùng hướng tuyệt đối = 1.0)")
    print(f"- Cosine(u, w):    {cosine_similarity(u, w):.4f} (Góc lệch lớn)")
    print(f"- Dot Product(u, v): {dot_product(u, v):.4f} (Bị ảnh hưởng bởi độ dài vector)")
    print(f"- Euclidean(u, v):   {euclidean_distance(u, v):.4f} (Khoảng cách hình học)")

class ExactVectorDB:
    def __init__(self):
        self.vectors = []
        self.texts = []

    def add(self, vector: np.ndarray, text: str):
        self.vectors.append(vector)
        self.texts.append(text)

    def search(self, query_vec: np.ndarray, top_k: int = 2):
        scores = [cosine_similarity(query_vec, v) for v in self.vectors]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.texts[i], scores[i]) for i in top_indices]

class IVFIndex:
    def __init__(self, k_clusters=2):
        self.k_clusters = k_clusters
        self.centroids = None
        self.inverted_lists = {} 

    def train_and_build(self, vectors: np.ndarray, texts: list):
        np.random.seed(42)
        indices = np.random.choice(len(vectors), self.k_clusters, replace=False)
        self.centroids = vectors[indices].copy()

        self.inverted_lists = {i: [] for i in range(self.k_clusters)}
        for v, t in zip(vectors, texts):
            dists = [euclidean_distance(v, c) for c in self.centroids]
            closest_cluster = int(np.argmin(dists))
            self.inverted_lists[closest_cluster].append((v, t))

    def search(self, query_vec: np.ndarray, nprobe: int = 1, top_k: int = 2):
        dists = [euclidean_distance(query_vec, c) for c in self.centroids]
        best_clusters = np.argsort(dists)[:nprobe]
        candidates = []
        for cluster_id in best_clusters:
            candidates.extend(self.inverted_lists[cluster_id])

        scores = [(text, cosine_similarity(query_vec, v)) for v, text in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

def demo_search_algorithms():
    docs = [
        "Chó là loài thú cưng rất trung thành",
        "Mèo thích bắt chuột và ngủ nhiều",
        "Python là ngôn ngữ số 1 cho AI và Machine Learning",
        "Lập trình hướng đối tượng trong C++",
        "Dự báo thời tiết mai có mưa rào"
    ]
    doc_vectors = np.array([
        [0.9, 0.1, 0.0],
        [0.85, 0.1, 0.05],
        [0.05, 0.95, 0.1],
        [0.0, 0.9, 0.05],
        [0.1, 0.05, 0.9]
    ])

    exact_db = ExactVectorDB()
    for v, t in zip(doc_vectors, docs):
        exact_db.add(v, t)

    query_vec = np.array([0.92, 0.08, 0.0])
    print("[Query] Vector tìm kiếm thuộc chủ đề Động vật:")
    
    print("\n-- Kết quả Exact Search (Quét toàn bộ): --")
    for doc, score in exact_db.search(query_vec, top_k=2):
        print(f"  + [{score:.4f}] {doc}")

    ivf = IVFIndex(k_clusters=2)
    ivf.train_and_build(doc_vectors, docs)
    print("\n-- Kết quả IVF Index (Chỉ quét cụm gần nhất - nprobe=1): --")
    for doc, score in ivf.search(query_vec, nprobe=1, top_k=2):
        print(f"  + [{score:.4f}] {doc}")

def chunk_text(text: str, chunk_size: int = 50, chunk_overlap: int = 15):
    chunks = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start += step
    return chunks

def demo_chunking():
    print("\n" + "=" * 60)
    print("3. DEMO CHUNKING: CHUNK SIZE & CHUNK OVERLAP")
    print("=" * 60)
    sample_text = (
        "Vector Database là cơ sở dữ liệu chuyên dụng để lưu trữ và tìm kiếm vector embedding. "
        "Nó đóng vai trò cốt lõi trong hệ thống RAG giúp mở rộng bộ nhớ dài hạn cho các mô hình AI LLM."
    )
    print(f"Văn bản gốc ({len(sample_text)} ký tự):\n\"{sample_text}\"\n")
    
    chunks = chunk_text(sample_text, chunk_size=60, chunk_overlap=20)
    print(f"Chia thành {len(chunks)} chunks (Size=60, Overlap=20):")
    for i, c in enumerate(chunks, 1):
        print(f"  Chunk {i} [{len(c)} ký tự]: \"{c}\"")

def demo_hybrid_search():
    print("\n" + "=" * 60)
    print("4. DEMO HYBRID SEARCH (KEYWORD + VECTOR) & RERANKING")
    print("=" * 60)

    corpus = [
        "iPhone 15 Pro Max 256GB chính hãng màu titan tự nhiên",
        "Điện thoại thông minh chụp ảnh đẹp nhất của Apple năm nay",
        "Máy tính bảng iPad Pro màn hình OLED siêu nét",
        "Hướng dẫn bảo hành điện thoại iPhone khi bị rơi vỡ"
    ]
    embeddings = np.array([
        [0.9, 0.2],
        [0.85, 0.7],
        [0.4, 0.3],
        [0.6, 0.1]
    ])

    query_str = "iPhone 15 Pro Max"
    query_vec = np.array([0.9, 0.2])

    print(f"Câu hỏi: \"{query_str}\"\n")

    keyword_scores = [1.0 if "iPhone 15 Pro Max" in doc else (0.5 if "iPhone" in doc else 0.0) for doc in corpus]
    vector_scores = [cosine_similarity(query_vec, v) for v in embeddings]
    hybrid_scores = [0.5 * k + 0.5 * v for k, v in zip(keyword_scores, vector_scores)]

    print(f"{'Văn bản':<55} | {'Keyword':<7} | {'Vector':<7} | {'Hybrid':<7}")
    print("-" * 85)
    for doc, k_s, v_s, h_s in zip(corpus, keyword_scores, vector_scores, hybrid_scores):
        print(f"{doc[:52] + '...':<55} | {k_s:<7.2f} | {v_s:<7.2f} | {h_s:<7.2f}")

if __name__ == "__main__":
    demo_metrics()
    demo_search_algorithms()
    demo_chunking()
    demo_hybrid_search()
