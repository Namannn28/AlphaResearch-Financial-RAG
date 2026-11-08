# Alex Chen
**Role**: Senior AI Systems Engineer
**Email**: alex.chen.ai@example.com | **Location**: San Francisco, CA | **GitHub**: github.com/alexchen-ai

---

## Professional Summary
Senior AI Systems Engineer with 6+ years of experience designing and scaling low-latency machine learning inference platforms, distributed model serving infrastructure, and LLM orchestration systems. Proven track record of reducing p99 inference latency by 45% using TensorRT-LLM and vLLM on Kubernetes GPU clusters.

---

## Technical Skills
- **Core Languages**: Python, C++, Go, SQL, Bash
- **ML & Systems**: PyTorch, TensorRT-LLM, vLLM, Triton Inference Server, Ray, Hugging Face Transformers
- **Infrastructure & Cloud**: Kubernetes (EKS), Docker, Terraform, AWS (EC2, S3, SageMaker), Helm, Prometheus, Grafana
- **Data & Vector Stores**: Milvus, Qdrant, Redis, PostgreSQL, Kafka
- **Evaluation & CI/CD**: Ragas, TruLens, MLflow, GitHub Actions

---

## Work Experience

### Staff Machine Learning Systems Engineer | ScaleFlow AI (2022 – Present)
- Architected a multi-tenant LLM serving gateway using vLLM and Triton Inference Server, handling over 25 million daily API calls with 99.98% uptime.
- Implemented speculative decoding and continuous batching techniques, decreasing average token latency from 52ms to 28ms per token.
- Engineered an automated semantic caching layer utilizing Redis and Milvus vector search, cutting OpenAI API operational costs by $42,000 monthly.
- Managed a fleet of 64 NVIDIA H100 GPUs across 8 Kubernetes clusters, authoring custom Kubernetes Device Plugins for dynamic GPU memory allocation.

### Senior Infrastructure Engineer | Datacore Systems (2019 – 2022)
- Built distributed training pipelines using PyTorch DDP and Ray Train on AWS EKS, accelerating recommendation model retraining cycles from 14 hours to 3.5 hours.
- Designed automated canary deployment strategies for 12 microservices with Argo Rollouts and Istio service mesh.
- Reduced cloud infrastructure spend by 28% through aggressive spot-instance provisioning and intelligent node termination handlers.

---

## Key Projects

### OpenServe-LLM (Open Source Core Contributor)
- Created an open-source benchmarking suite for measuring KV-cache memory fragmentation across quantized models (AWQ, GPTQ, FP8).
- Implemented asynchronous batch streaming over HTTP/2 and gRPC protocols, adopted by over 1,200 GitHub stars.

### Dynamic Context-Aware RAG Pipeline
- Implemented a hybrid retrieval engine combining BM25 lexical ranking and dense vector search (BGE-Large) with a Cross-Encoder reranker.
- Evaluated retrieval fidelity using Hit Rate@5 (89.2%) and NDCG@10 (0.84) across 500 domain-specific technical queries.

---

## Education
- **Master of Science in Computer Science** (Specialization in Distributed Systems) | Stanford University (2017 – 2019)
- **Bachelor of Science in Computer Engineering** | University of Washington (2013 – 2017)
