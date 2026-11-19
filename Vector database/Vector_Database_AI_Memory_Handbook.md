# SỔ TAY HỎI ĐÁP: VECTOR DATABASE & AI MEMORY

---

## MỤC LỤC
- [1. Vector & Vector Database](#1-vector--vector-database)
  - [Vector là gì? Vector được sử dụng như thế nào trong AI/ML?](#vector-là-gì-vector-được-sử-dụng-như-thế-nào-trong-aiml)
  - [Embedding là gì? Quá trình chuyển đổi dữ liệu thành vector diễn ra như thế nào?](#embedding-là-gì-quá-trình-chuyển-đổi-dữ-liệu-thành-vector-diễn-ra-như-thế-nào)
  - [Vector Database là gì? Khác biệt với Database truyền thống?](#vector-database-là-gì-khác-biệt-với-database-truyền-thống)
  - [Các phép đo độ tương đồng giữa vector (Cosine, Dot Product, Euclidean)](#các-phép-đo-độ-tương-đồng-giữa-vector-cosine-similarity-dot-product-euclidean-distance)
- [2. Vector Database Search Algorithms](#2-vector-database-search-algorithms)
  - [Các phương pháp tìm kiếm vector phổ biến](#các-phương-pháp-tìm-kiếm-vector-phổ-biến)
  - [Exact Search và Approximate Nearest Neighbor (ANN)](#exact-search-và-approximate-nearest-neighbor-ann)
  - [Nguyên lý hoạt động cơ bản của HNSW, IVF và PQ](#nguyên-lý-hoạt-động-cơ-bản-của-hnsw-ivf-và-pq)
  - [Ưu nhược điểm và trường hợp sử dụng của từng phương pháp](#ưu-nhược-điểm-và-trường-hợp-sử-dụng-của-từng-phương-pháp)
- [3. AI Memory Management](#3-ai-memory-management)
  - [AI Memory là gì? Cách AI quản lý và sử dụng thông tin](#ai-memory-là-gì-cách-ai-quản-lý-và-sử-dụng-thông-tin)
  - [Short-term Memory và Long-term Memory](#short-term-memory-và-long-term-memory)
  - [Context Window, Conversation History và Memory khác nhau như thế nào?](#context-window-conversation-history-và-memory-khác-nhau-như-thế-nào)
  - [Các phương pháp lưu trữ, truy xuất và cập nhật Memory](#các-phương-pháp-lưu-trữ-truy-xuất-và-cập-nhật-memory)
- [4. Chunking Strategies](#4-chunking-strategies)
  - [Chunking là gì? Tại sao cần Chunking?](#chunking-là-gì-tại-sao-cần-chunking)
  - [Các phương pháp Chunking phổ biến](#các-phương-pháp-chunking-phổ-biến)
  - [Chunk Size và Chunk Overlap](#chunk-size-và-chunk-overlap)
  - [Ảnh hưởng của chiến lược Chunking đến chất lượng Retrieval](#ảnh-hưởng-của-chiến-lược-chunking-đến-chất-lượng-retrieval)
- [5. Information Analysis & Synthesis](#5-information-analysis--synthesis)
  - [Các phương pháp phân tích và trích xuất thông tin từ dữ liệu](#các-phương-pháp-phân-tích-và-trích-xuất-thông-tin-từ-dữ-liệu)
  - [Cách tổ chức, phân loại và tổng hợp thông tin từ nhiều nguồn](#cách-tổ-chức-phân-loại-và-tổng-hợp-thông-tin-từ-nhiều-nguồn)
  - [Keyword Search và Semantic Search](#keyword-search-và-semantic-search)
  - [Hybrid Search và Reranking](#hybrid-search-và-reranking)
  - [Các vấn đề khi xử lý thông tin trùng lặp, mâu thuẫn hoặc thiếu ngữ cảnh](#các-vấn-đề-khi-xử-lý-thông-tin-trùng-lặp-mâu-thuẫn-hoặc-thiếu-ngữ-cảnh)

---

# 1. Vector & Vector Database

### Vector là gì? Vector được sử dụng như thế nào trong AI/ML?
- **Định nghĩa:** Về mặt toán học, **Vector** là một mảng số thực có thứ tự trong không gian nhiều chiều: 
  $$\mathbf{v} = [v_1, v_2, \dots, v_d] \in \mathbb{R}^d$$ 
  Trong đó $d$ là số chiều (dimensions). Trong AI hiện đại, vector thường có từ vài trăm đến hàng nghìn chiều (ví dụ: OpenAI `text-embedding-3-small` là 1536 chiều).
- **Cách sử dụng trong AI/ML:**
  1. **Biểu diễn đặc trưng (Feature Representation):** Chuyển đổi mọi loại dữ liệu thô (chữ viết, điểm ảnh, sóng âm) thành vector để mạng nơ-ron có thể hiểu và tính toán.
  2. **Lưu trữ tri thức:** Toàn bộ trọng số (weights) và trạng thái ẩn (hidden states) của các mô hình AI/LLM đều là vector và ma trận.
  3. **So sánh tương đồng ngữ nghĩa (Similarity Search):** Quy bài toán so sánh ý nghĩa ngôn ngữ về bài toán đo khoảng cách hình học giữa các điểm vector trong không gian.

---

### Embedding là gì? Quá trình chuyển đổi dữ liệu thành vector diễn ra như thế nào?
- **Embedding:** Là kỹ thuật ánh xạ một đối tượng (từ, câu, hình ảnh) thành một vector số thực liên tục sao cho **những đối tượng có ý nghĩa tương đồng sẽ nằm gần nhau trong không gian vector**.
- **Quá trình chuyển đổi văn bản thành vector (4 bước):**
  1. **Tokenization:** Văn bản thô được bẻ nhỏ thành chuỗi token và ánh xạ thành dãy số nguyên (Token IDs).
  2. **Lookup & Positional Encoding:** Mỗi token được gắn một vector ban đầu và cộng thêm vector vị trí để mô hình nhận diện thứ tự từ.
  3. **Tương tác ngữ cảnh (Self-Attention qua Transformer):** Các từ "giao tiếp" và hấp thu ngữ cảnh xung quanh để tạo ra biểu diễn chính xác (từ "ngân hàng" trong kinh tế sẽ có vector khác "ngân hàng" bờ sông).
  4. **Pooling:** Gom toàn bộ ma trận các vector token thành **1 vector duy nhất** đại diện cho cả đoạn (qua `Mean Pooling` hoặc lấy vector của token `[CLS]`), sau đó thường chuẩn hóa về độ dài $L_2$ norm = 1.

---

### Vector Database là gì? Khác biệt với Database truyền thống?
- **Vector Database:** Là hệ quản trị cơ sở dữ liệu chuyên biệt để lưu trữ, lập chỉ mục (indexing) và tìm kiếm độ tương đồng trên hàng triệu đến hàng tỷ vector kèm metadata với độ trễ tính bằng mili-giây.

| Tiêu chí | Vector Database (Pinecone, Milvus, Qdrant) | Database truyền thống (PostgreSQL, MySQL, MongoDB) |
| :--- | :--- | :--- |
| **Dữ liệu chính** | Vector đa chiều kèm metadata | Dữ liệu dạng bảng (hàng/cột), JSON, Key-Value |
| **Cách truy vấn** | **Tìm kiếm tương đồng ngữ nghĩa** (Cosine, Dot, Euclidean) | **Khớp chính xác** (`WHERE id = 5`, `LIKE '%abc%'`, `JOIN`) |
| **Cơ chế Index** | Thuật toán xấp xỉ ANN (HNSW, IVF, PQ, ScaNN) | Cấu trúc cây/băm (B-Tree, B+Tree, Hash Index) |
| **Loại dữ liệu mạnh** | Dữ liệu phi cấu trúc (văn bản, ảnh, âm thanh, video) | Dữ liệu có cấu trúc hoặc bán cấu trúc rõ ràng |
| **Ứng dụng điển hình** | RAG cho AI, tìm kiếm ngữ nghĩa, nhận diện ảnh/khuôn mặt | Giao dịch tài chính (ACID), quản lý đơn hàng, hồ sơ người dùng |

---

### Các phép đo độ tương đồng giữa vector (Cosine Similarity, Dot Product, Euclidean Distance)

Cho hai vector $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$:

1. **Cosine Similarity (Độ tương đồng Cosine):**
   - Đo **góc lệch** giữa hai vector, không phụ thuộc vào độ dài (độ lớn vector).
   - Công thức: 
     $$\text{Cosine}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \frac{\sum u_i v_i}{\sqrt{\sum u_i^2}\sqrt{\sum v_i^2}}$$
   - Khoảng giá trị: $[-1, 1]$ (1 là hoàn toàn trùng hướng/ý nghĩa).
   - *Ứng dụng:* Tốt nhất cho so sánh văn bản (Text Embeddings) vì độ dài câu văn không làm lệch ý nghĩa.

2. **Dot Product (Tích vô hướng / Inner Product):**
   - Đo cả **góc lệch lẫn độ lớn (magnitude)** của vector.
   - Công thức: 
     $$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^d u_i v_i$$
   - *Ứng dụng:* Khi vector đã được chuẩn hóa độ dài về 1 ($\|\mathbf{u}\| = \|\mathbf{v}\| = 1$), Dot Product cho kết quả bằng Cosine Similarity nhưng tính toán nhanh hơn nhiều.

3. **Euclidean Distance (Khoảng cách Euclid / $L_2$ Distance):**
   - Đo **khoảng cách đường thẳng hình học** giữa hai điểm đầu mút vector.
   - Công thức: 
     $$d(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{i=1}^d (u_i - v_i)^2}$$
   - Khoảng cách càng nhỏ thì hai đối tượng càng giống nhau ($d = 0$ là hoàn toàn trùng nhau).
   - *Ứng dụng:* Xử lý ảnh, âm thanh hoặc dữ liệu mà độ lớn của đặc trưng có ý nghĩa quan trọng.

---

# 2. Vector Database Search Algorithms

### Các phương pháp tìm kiếm vector phổ biến
- **Tree-based (KD-Tree, Annoy):** Phân chia không gian bằng các siêu phẳng (hyperplanes) thành cấu trúc cây nhị phân.
- **Inverted File (IVF):** Phân cụm không gian thành các vùng Voronoi cells.
- **Graph-based (HNSW):** Xây dựng mạng lưới liên kết láng giềng nhiều tầng.
- **Quantization-based (PQ, SCaNN):** Nén dữ liệu vector thành các mã chỉ mục nhỏ để tiết kiệm RAM.

---

### Exact Search và Approximate Nearest Neighbor (ANN)

| Tiêu chí | Exact Search (k-NN / Brute-force / Flat) | Approximate Nearest Neighbor (ANN) |
| :--- | :--- | :--- |
| **Nguyên lý** | Đo khoảng cách từ truy vấn tới **toàn bộ** vector trong kho | Dùng cấu trúc đồ thị, phân cụm hoặc băm để chỉ duyệt vùng tiềm năng |
| **Độ chính xác (Recall)** | 100% (Hoàn hảo) | 95% – 99% (Chấp nhận sai số nhỏ) |
| **Độ phức tạp** | $O(N \cdot d)$ (Tăng tuyến tính theo số lượng bản ghi) | $O(\log N)$ hoặc $O(1)$ |
| **Tốc độ trên data lớn** | Rất chậm khi $N > 1.000.000$ | Cực nhanh (chỉ mất vài mili-giây) |

---

### Nguyên lý hoạt động cơ bản của HNSW, IVF và PQ

1. **HNSW (Hierarchical Navigable Small World):**
   - **Nguyên lý:** Xây dựng mạng lưới đồ thị nhiều tầng (tương tự ý tưởng Skip-List).
   - **Cách tìm:** Bắt đầu từ tầng cao nhất (thưa nhất) để "nhảy" những bước xa đến gần khu vực mục tiêu, sau đó tụt dần xuống các tầng thấp hơn (dày đặc hơn) để dò tìm láng giềng gần nhất.

2. **IVF (Inverted File Index):**
   - **Nguyên lý:** Dùng thuật toán $K$-Means chia không gian vector thành $k$ cụm (Voronoi cells).
   - **Cách tìm:** Khi tìm kiếm, chỉ xác định vài tâm cụm ($Centroids$) gần vector truy vấn nhất (thông qua tham số `nprobe`), rồi chỉ quét các vector bên trong các cụm đó thay vì quét toàn bộ database.

3. **PQ (Product Quantization):**
   - **Nguyên lý:** Nén vector. Chia vector $d$ chiều thành $m$ phần nhỏ (sub-vectors). Với mỗi phần nhỏ, áp dụng phân cụm để ánh xạ nó về một giá trị mã centroid (Codebook).
   - **Kết quả:** Biến vector số thực 32-bit (FP32) cồng kềnh thành chuỗi vài byte ngắn, giảm dung lượng bộ nhớ từ 4x đến 16x.

---

### Ưu nhược điểm và trường hợp sử dụng của từng phương pháp

| Phương pháp | Ưu điểm | Nhược điểm | Trường hợp sử dụng phù hợp |
| :--- | :--- | :--- | :--- |
| **HNSW** | Tốc độ truy vấn siêu nhanh, Recall cực cao (98–99%) | Tốn nhiều RAM nhất, build index tốn thời gian và CPU | Hệ thống yêu cầu độ trễ thấp (< 10ms), độ chính xác cao và có ngân sách RAM dồi dào. |
| **IVF** | Tiết kiệm RAM hơn HNSW, tốc độ tạo index nhanh | Phải tuning kỹ tham số `nlist`, `nprobe` để cân bằng tốc độ và Recall | Dữ liệu quy mô lớn, tài nguyên máy chủ bộ nhớ vừa phải. |
| **PQ / IVF-PQ** | Tiết kiệm 75–90% RAM, chứa được hàng trăm triệu vector trên 1 máy | Giảm độ chính xác (mất mát dữ liệu do nén) | Database quy mô cực lớn (hàng chục đến hàng trăm triệu vector), chi phí phần cứng giới hạn. |

---

# 3. AI Memory Management

### AI Memory là gì? Cách AI quản lý và sử dụng thông tin
- **AI Memory (Bộ nhớ AI):** Là cơ chế lưu trữ, quản lý và khôi phục ngữ cảnh hoặc dữ liệu cá nhân hóa qua các phiên hội thoại, giúp AI duy trì trải nghiệm liền mạch mà không cần huấn luyện lại mô hình (fine-tuning).
- **Cách quản lý và sử dụng:**
  $$\text{Hội thoại} \longrightarrow \text{Trích xuất thông tin cốt lõi} \longrightarrow \text{Lưu vào kho nhớ (DB/Cache)} \longrightarrow \text{Truy xuất & Nạp vào Prompt khi cần}$$

---

### Short-term Memory và Long-term Memory

```
┌────────────────────────────────────────────────────────┐
│                      AI SYSTEM                         │
│  ┌────────────────────────┐  ┌───────────────────────┐ │
│  │   Short-term Memory    │  │   Long-term Memory    │ │
│  │ (RAM: Lượt chat hiện tại)│  │(Disk/DB: Tuần/Tháng/Năm)│ │
│  │ Context Window / Cache │  │ Vector DB / Entity DB │ │
│  └────────────────────────┘  └───────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

- **Short-term Memory (Bộ nhớ ngắn hạn):**
  - Chứa thông tin ngữ cảnh tức thời trong phiên trò chuyện đang diễn ra (vài câu hỏi - đáp gần nhất).
  - Tồn tại trong bộ nhớ tạm thời (RAM / Context Window) và biến mất khi người dùng đóng phiên làm việc.
- **Long-term Memory (Bộ nhớ dài hạn):**
  - Lưu trữ sở thích cá nhân, sự kiện, kiến thức xuyên suốt qua nhiều ngày, tháng hoặc năm.
  - Được lưu bền vững vào cơ sở dữ liệu (Vector DB, Graph DB, Relational DB) và chỉ truy xuất các mảnh liên quan khi cần.

---

### Context Window, Conversation History và Memory khác nhau như thế nào?

| Khái niệm | Bản chất | Đặc điểm / Phạm vi |
| :--- | :--- | :--- |
| **Context Window** | Giới hạn dung lượng token tối đa mà LLM có thể tiếp nhận trong một lượt suy luận | Giới hạn phần cứng/mô hình cố định (ví dụ: 128k, 1M token). Vượt quá sẽ bị lỗi hoặc tràn bộ nhớ. |
| **Conversation History** | Toàn bộ nhật ký hội thoại dạng văn bản thô giữa người dùng và trợ lý | Dài vô tận theo thời gian; nếu đưa hết vào Context Window sẽ làm chậm hệ thống và tốn chi phí. |
| **Memory** | Thông tin **đã qua xử lý, trích xuất và cô đọng** từ lịch sử hội thoại | Tồn tại độc lập; được chọn lọc có chủ đích để nạp một lượng vừa đủ vào Context Window. |

---

### Các phương pháp lưu trữ, truy xuất và cập nhật Memory
- **Lưu trữ (Storage):**
  - *Vector Memory:* Lưu đoạn trích ngắn dưới dạng vector embedding để tìm lại bằng độ tương đồng.
  - *Entity/Graph Memory:* Lưu các thực thể và mối quan hệ (ví dụ: `User` $\to$ `sở thích` $\to$ `đọc sách`).
  - *Summary Buffer:* Tóm tắt định kỳ các lượt chat cũ thành một đoạn văn ngắn gọn.
- **Truy xuất (Retrieval):**
  - Kết hợp độ tương đồng ngữ nghĩa (Relevance), thời gian xuất hiện (Recency), và mức độ quan trọng (Importance).
- **Cập nhật (Update):**
  - *Consolidation:* Gom các mẩu ký ức vụn vặt thành một kết luận khái quát.
  - *Conflict Resolution:* Khi thông tin mới mâu thuẫn thông tin cũ, hệ thống sẽ ghi đè hoặc hạ độ ưu tiên của dữ liệu cũ.

---

# 4. Chunking Strategies

### Chunking là gì? Tại sao cần Chunking?
- **Chunking:** Là kỹ thuật chia một tài liệu lớn (sách, báo cáo, bài báo) thành nhiều đoạn văn bản nhỏ hơn (chunks) trước khi tạo embedding và đưa vào cơ sở dữ liệu.
- **Tại sao cần Chunking?**
  1. **Giới hạn của Embedding Model:** Hầu hết mô hình embedding chỉ xử lý tối đa 512 hoặc 8192 token mỗi lần.
  2. **Tránh loãng ngữ nghĩa (Semantic Dilution):** Một đoạn văn quá dài sẽ chứa nhiều chủ đề khác nhau, khiến vector đại diện bị nhiễu và làm giảm độ chuẩn xác khi tìm kiếm.
  3. **Tối ưu hóa phản hồi của LLM:** Cung cấp đúng đoạn văn chứa câu trả lời thay vì nạp cả cuốn sách vài trăm trang vào prompt.

---

### Các phương pháp Chunking phổ biến
1. **Fixed-size Chunking (Cắt theo kích thước cố định):**
   - Cắt theo số lượng ký tự hoặc token cố định (ví dụ: cứ 500 ký tự cắt 1 đoạn).
   - *Đánh giá:* Rất đơn giản nhưng dễ làm đứt gãy câu và ngữ nghĩa.
2. **Recursive Character Chunking (Cắt đệ quy theo dấu ngắt tự nhiên):**
   - Cố gắng ngắt ở các ranh giới tự nhiên theo thứ tự: hết đoạn (`\n\n`) $\to$ hết dòng (`\n`) $\to$ hết câu (`. `) $\to$ từ (` `).
   - *Đánh giá:* Phương pháp chuẩn và tối ưu nhất cho văn bản thông thường trong các hệ thống RAG hiện nay.
3. **Document-based / Structural Chunking (Cắt theo cấu trúc):**
   - Cắt dựa trên tiêu đề Markdown (`#`, `##`), thẻ HTML (`<h1>`, `<p>`), cấu trúc bảng hoặc file JSON/code.
4. **Semantic Chunking (Cắt theo ngữ nghĩa):**
   - Đo khoảng cách ngữ nghĩa giữa các câu liên tiếp bằng embedding. Khi có sự chênh lệch lớn (chuyển đổi chủ đề), hệ thống sẽ ngắt chunk.

---

### Chunk Size và Chunk Overlap

```
Văn bản gốc:  [======== Chunk 1 (Size: 500) ========]
                                   [== Overlap: 100 ==]
                                   [======== Chunk 2 (Size: 500) ========]
```

- **Chunk Size (Kích thước đoạn):**
  - *Nhỏ (100–300 token):* Tìm kiếm cực kỳ chính xác vào tiểu tiết, nhưng thiếu ngữ cảnh bao quát.
  - *Lớn (800–1500 token):* Ngữ cảnh đầy đủ, nhưng vector dễ bị pha loãng và giảm độ khớp khi tìm kiếm.
- **Chunk Overlap (Độ gối đầu):**
  - Là phần văn bản ở cuối chunk trước được lặp lại ở đầu chunk sau (thường từ 10% – 20% Chunk Size).
  - *Mục đích:* Đảm bảo các ý tưởng hoặc thông tin nằm ngay ranh giới cắt không bị đứt đoạn hay mất bối cảnh.

---

### Ảnh hưởng của chiến lược Chunking đến chất lượng Retrieval
- **Chunk quá lớn:** Vector phản ánh giá trị trung bình của nhiều ý tưởng hỗn tạp $\to$ điểm tương đồng (Similarity score) bị kéo thấp, dễ bỏ sót dữ liệu.
- **Chunk quá nhỏ:** Trả về các mẩu vụn thông tin không đủ ý $\to$ LLM không hiểu trọn vẹn vấn đề và dễ sinh ra hiện tượng ảo giác (Hallucination).
- *Chiến lược tối ưu:* Dùng kỹ thuật **Parent-Document Retrieval** (truy vấn tìm kiếm trên Small Chunks để đảm bảo độ chính xác cao, nhưng khi trả kết quả cho LLM thì trả về cả Parent Chunk lớn chứa nó).

---

# 5. Information Analysis & Synthesis

### Các phương pháp phân tích và trích xuất thông tin từ dữ liệu
- **Named Entity Recognition (NER):** Nhận diện và bóc tách các thực thể cụ thể (tên người, tổ chức, địa điểm, số liệu, ngày tháng).
- **Relation Extraction:** Trích xuất mối quan hệ giữa các thực thể (ví dụ: `[A] là CEO của [B]`).
- **Structured Extraction via LLM:** Yêu cầu LLM đọc dữ liệu phi cấu trúc và xuất ra định dạng JSON/Pydantic schema chặt chẽ.
- **Text Summarization:** Tóm tắt văn bản theo dạng trích xuất câu gốc (extractive) hoặc diễn đạt lại bằng từ ngữ mới (abstractive).

---

### Cách tổ chức, phân loại và tổng hợp thông tin từ nhiều nguồn
1. **Chuẩn hóa (Normalization):** Chuyển đổi mọi định dạng nguồn (PDF, Word, Web, SQL) về văn bản sạch đồng nhất (thường là Markdown).
2. **Bổ sung Metadata:** Gắn nhãn nhất quán cho từng tài liệu: nguồn (URL/file), tác giả, ngày cập nhật, danh mục, quyền truy cập.
3. **Cấu trúc phân tầng (Hierarchical Organization):** Nhóm các tài liệu theo chủ đề tổng quan $\to$ chi tiết để dễ dàng khoanh vùng khi tìm kiếm.
4. **Quy trình tổng hợp Map-Reduce:** Chia nhỏ việc đọc và tóm tắt từng nguồn tài liệu cho các agent/LLM chạy song song, sau đó tổng hợp các kết quả tóm tắt lại thành một báo cáo duy nhất.

---

### Keyword Search và Semantic Search

| Tiêu chí | Keyword Search (BM25, TF-IDF) | Semantic Search (Vector Embedding) |
| :--- | :--- | :--- |
| **Nguyên lý** | Đếm tần suất xuất hiện và đối khớp các từ khóa chính xác | Đo khoảng cách góc giữa các vector ngữ nghĩa trong không gian đa chiều |
| **Điểm mạnh** | Tìm kiếm cực chuẩn mã sản phẩm, thuật ngữ hiếm, số ID, tên riêng (ví dụ: `iPhone 15 Pro`, mã lỗi `ERR_007`) | Hiểu từ đồng nghĩa, câu hỏi ẩn ý, không yêu cầu người dùng phải nhớ chính xác từ khóa |
| **Điểm yếu** | Hoàn toàn không hiểu ngữ cảnh và từ đồng nghĩa; dễ bị từ đa nghĩa gây nhiễu | Kém khi tìm kiếm chính xác các từ hiếm hoặc chuỗi ký tự đặc biệt |

---

### Hybrid Search và Reranking

```
User Query ──┬──> [BM25 Search]     ──> Top 50 kết quả ──┐
             │                                           ├──> [RRF / Weighted Score] ──> Top 50 ──> [Cross-Encoder Reranker] ──> Top 5 cuối cùng
             └──> [Vector Search]   ──> Top 50 kết quả ──┘
```

- **Hybrid Search (Tìm kiếm lai):** Chạy đồng thời cả **Keyword Search (BM25)** và **Vector Search (Semantic)**, sau đó kết hợp danh sách ứng viên lại bằng thuật toán **Reciprocal Rank Fusion (RRF)**. Đảm bảo không bỏ sót cả từ khóa hiếm lẫn ngữ nghĩa trừu tượng.
- **Reranking (Xếp hạng lại):**
  - *Bước 1 (Retrieval nhanh):* Dùng Bi-Encoder / Vector DB để kéo về Top 50–100 kết quả có điểm số cao nhất trong vài mili-giây.
  - *Bước 2 (Chấm điểm chi tiết):* Đưa từng cặp `(Query, Document)` qua mô hình Cross-Encoder chuyên dụng (như Cohere Rerank, BGE-Reranker) để đánh giá độ liên quan ngữ nghĩa sâu sắc, chọn ra Top 3–5 tài liệu thực sự tốt nhất cho LLM.

---

### Các vấn đề khi xử lý thông tin trùng lặp, mâu thuẫn hoặc thiếu ngữ cảnh

1. **Thông tin trùng lặp (Redundancy):**
   - *Vấn đề:* Lãng phí context window, làm câu trả lời bị lặp ý.
   - *Giải pháp:* Dùng ngưỡng lọc độ tương đồng (Cosine threshold > 0.9) hoặc giải thuật **Maximal Marginal Relevance (MMR)** để chọn các tài liệu vừa liên quan vừa đa dạng thông tin.

2. **Thông tin mâu thuẫn (Contradiction):**
   - *Vấn đề:* Hai tài liệu cung cấp số liệu hoặc nhận định trái ngược nhau, khiến AI đưa ra câu trả lời sai lệch.
   - *Giải pháp:*
     - Lọc theo thời gian (Recency / Temporal Decay): Ưu tiên tài liệu mới nhất.
     - Lọc theo độ uy tín nguồn (Source Authority): Gán trọng số cao hơn cho tài liệu chính thống.
     - Prompt Engineering: Nhắc nhở mô hình chủ động chỉ ra các điểm khác biệt giữa các nguồn khi phát hiện mâu thuẫn.

3. **Thông tin thiếu ngữ cảnh (Context Fragmentation):**
   - *Vấn đề:* Đoạn cắt chỉ chứa kết luận cộc lốc như "Doanh thu tăng 25%" mà không nêu rõ năm nào hay của công ty nào.
   - *Giải pháp:*
     - **Contextual Chunking:** Cho LLM bổ sung 1–2 câu tóm tắt ngữ cảnh toàn văn vào đầu mỗi chunk trước khi tạo embedding.
     - **Parent-Document Retrieval:** Tìm kiếm trên chunk nhỏ nhưng nạp toàn bộ đoạn văn/trang lớn chứa nó vào Prompt.
