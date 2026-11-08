"""
Generates 105+ diverse, realistic, industry-aligned Job Descriptions
for technical candidate auditing in CandidateFit-RAG.
"""

from pathlib import Path
import json

JOB_DESCRIPTIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "job_descriptions"
JOB_DESCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Catalog of 105 distinct technical positions across 7 industry categories
ROLES_CATALOG = [
    # --- Category 1: Software Engineering (20 Roles) ---
    {
        "id": "frontend_react_engineer",
        "title": "Senior Frontend Engineer (React/TypeScript)",
        "dept": "Frontend Engineering",
        "company": "Veloce Technologies",
        "location": "San Francisco, CA / Remote",
        "overview": "Build accessible, pixel-perfect, high-performance web applications using React 18+, TypeScript, and Tailwind CSS. Focus on Core Web Vitals and component reusability.",
        "responsibilities": [
            "Architect and maintain reusable React component libraries compliant with WCAG 2.1 AA accessibility standards.",
            "Optimize client-side runtime performance, code-splitting, bundle sizes, and Core Web Vitals (LCP, FID, CLS).",
            "Partner with UX designers to translate complex Figma workflows into responsive, state-driven interfaces.",
            "Write comprehensive unit tests with Jest/React Testing Library and visual regression tests with Playwright."
        ],
        "requirements": [
            "4+ years of modern JavaScript and TypeScript experience with deep proficiency in React.",
            "Demonstrated experience with state management (Redux Toolkit, Zustand, or TanStack Query).",
            "Expertise in modern CSS architectures, Tailwind CSS, and headless UI libraries.",
            "Solid track record of web performance profiling using Chrome DevTools and Lighthouse."
        ]
    },
    {
        "id": "backend_go_microservices",
        "title": "Senior Backend Go Engineer (Distributed Systems)",
        "dept": "Platform Engineering",
        "company": "Aura Scale Systems",
        "location": "Austin, TX / Remote",
        "overview": "Engineer ultra-low-latency backend microservices in Go powering high-throughput transaction processing with gRPC and Kafka.",
        "responsibilities": [
            "Design and deploy concurrent Go services processing upwards of 50,000 requests per second.",
            "Implement resilient inter-service RPC communication using gRPC, Protocol Buffers, and Envoy proxy.",
            "Profile and eliminate memory leaks and garbage collection pauses using pprof and trace tooling.",
            "Ensure strict observability with distributed tracing via OpenTelemetry and Grafana Tempo."
        ],
        "requirements": [
            "4+ years of production experience building backend services in Go (Golang).",
            "Strong understanding of goroutines, channels, memory models, and concurrency synchronization.",
            "Proven experience with event streams using Apache Kafka or AWS Kinesis.",
            "Deep familiarity with PostgreSQL schema design, indexing, and connection pooling."
        ]
    },
    {
        "id": "fullstack_nextjs_engineer",
        "title": "Lead Full-Stack Next.js Architect",
        "dept": "Core Web Products",
        "company": "Pulse Commerce",
        "location": "Seattle, WA / Remote",
        "overview": "Lead the modernization of customer-facing storefronts with Next.js 14 App Router, Server Components, and GraphQL federation.",
        "responsibilities": [
            "Architect full-stack web applications utilizing Next.js 14 App Router, Server Actions, and React Server Components.",
            "Federate backend microservices into a unified GraphQL API gateway using Apollo Federation.",
            "Lead sprint architectural reviews, mentor engineers, and enforce type-safety from database to client.",
            "Implement edge-caching strategies using Cloudflare Workers and Vercel Edge Middleware."
        ],
        "requirements": [
            "5+ years of full-stack engineering with TypeScript, Node.js, and modern React/Next.js.",
            "Hands-on expertise with GraphQL schema design, resolvers, and caching patterns.",
            "Experience with PostgreSQL and ORM layers such as Prisma or Drizzle.",
            "Demonstrated leadership running agile sprint cycles and mentoring mid-level software engineers."
        ]
    },
    {
        "id": "mobile_ios_swift_engineer",
        "title": "Senior iOS Engineer (Swift & SwiftUI)",
        "dept": "Mobile Engineering",
        "company": "NeuraMobile",
        "location": "New York, NY / Remote",
        "overview": "Craft intuitive native iOS applications leveraging Swift, SwiftUI, Combine, and modern modular architectures.",
        "responsibilities": [
            "Develop modern, declarative native iOS interfaces using SwiftUI and Combine/Async-Await.",
            "Implement offline-first client synchronization using CoreData / SwiftData and BackgroundTasks.",
            "Ensure robust CI/CD delivery pipelines using Fastlane, TestFlight, and GitHub Actions.",
            "Profile memory footprints, frame rendering rates (ProMotion 120Hz), and energy efficiency with Instruments."
        ],
        "requirements": [
            "4+ years of dedicated native iOS app development with Swift.",
            "Published apps on the Apple App Store with demonstrated high ratings.",
            "Deep knowledge of Apple Human Interface Guidelines and modern iOS architecture (MVVM, Clean Swift).",
            "Experience with native unit testing using XCTest and UI automation."
        ]
    },
    {
        "id": "mobile_android_kotlin_engineer",
        "title": "Senior Android Engineer (Kotlin & Jetpack Compose)",
        "dept": "Mobile Engineering",
        "company": "NeuraMobile",
        "location": "Chicago, IL / Remote",
        "overview": "Build robust, scalable native Android applications using Kotlin, Jetpack Compose, Coroutines, and Flow.",
        "responsibilities": [
            "Architect responsive Android screens with Jetpack Compose following Material Design 3 guidelines.",
            "Manage asynchronous background workloads and state flow with Kotlin Coroutines and StateFlow.",
            "Implement local caching and room database migrations with SQLite / Room ORM.",
            "Conduct automated UI testing using Espresso and Compose Testing frameworks."
        ],
        "requirements": [
            "4+ years of native Android software development using modern Kotlin.",
            "Strong command of Jetpack Compose, Hilt / Koin dependency injection, and Android Jetpack libraries.",
            "Understanding of Android memory management, battery optimization, and lifecycle components.",
            "Experience managing release cycles on the Google Play Console."
        ]
    },
    {
        "id": "embedded_firmware_engineer",
        "title": "Embedded Systems & Firmware Engineer (C/C++)",
        "dept": "Hardware Systems",
        "company": "Apex Robotics Corp",
        "location": "San Jose, CA",
        "overview": "Develop deterministic real-time firmware for ARM Cortex-M microcontrollers and IoT edge sensory nodes.",
        "responsibilities": [
            "Author bare-metal and RTOS-based (FreeRTOS) C/C++ firmware for sensory acquisition and motor control.",
            "Implement hardware peripheral communication buses including I2C, SPI, UART, and CAN bus.",
            "Debug low-level hardware-firmware interactions using logic analyzers, oscilloscopes, and JTAG/SWD.",
            "Implement over-the-air (OTA) secure bootloaders with cryptographic firmware image signing."
        ],
        "requirements": [
            "3+ years of professional embedded C/C++ programming on ARM Cortex-M or RISC-V platforms.",
            "Hands-on experience with RTOS concepts: task scheduling, queues, semaphores, and interrupt service routines (ISRs).",
            "Ability to read hardware schematics, PCB layouts, and component datasheets.",
            "Familiarity with low-power battery-operated system optimization."
        ]
    },
    {
        "id": "rust_systems_engineer",
        "title": "Systems Software Engineer (Rust)",
        "dept": "Core Systems",
        "company": "Ironclad Infrastructure",
        "location": "Boston, MA / Remote",
        "overview": "Build memory-safe, zero-cost abstraction high-performance networking daemons and storage engines in Rust.",
        "responsibilities": [
            "Design and build concurrent storage and proxy daemons with async Rust using Tokio and Hyper.",
            "Profile systems to minimize memory allocations, CPU cache misses, and context switching.",
            "Write robust FFI bindings between Rust and existing C/C++ libraries.",
            "Participate in rigorous peer code reviews emphasizing memory safety, lifetime ergonomics, and panic-freedom."
        ],
        "requirements": [
            "3+ years of professional programming in Rust (ownership, lifetimes, traits, async/await).",
            "Strong foundation in operating system primitives, POSIX syscalls, and network sockets (TCP/UDP).",
            "Experience with benchmark profiling using Criterion.rs, perf, and flamegraphs.",
            "Passion for building bulletproof, panic-free infrastructure."
        ]
    },
    {
        "id": "python_fastapi_backend",
        "title": "Senior Python Backend Engineer (FastAPI & AsyncIO)",
        "dept": "Backend Services",
        "company": "Synthetix Cloud",
        "location": "Denver, CO / Remote",
        "overview": "Develop high-speed RESTful and WebSocket APIs powering SaaS applications using Python 3.12, FastAPI, and Redis.",
        "responsibilities": [
            "Design, document, and deploy production APIs using FastAPI and Pydantic validation.",
            "Implement real-time bidirectional messaging features with WebSockets and Redis Pub/Sub.",
            "Optimize asynchronous database queries with SQLAlchemy 2.0 (asyncio) and PostgreSQL.",
            "Manage background worker queues using Celery, RQ, or ARQ."
        ],
        "requirements": [
            "4+ years of Python engineering experience with focus on modern async frameworks (FastAPI, Starlette).",
            "Deep understanding of Python AsyncIO event loops, tasks, and concurrency boundaries.",
            "Strong relational database background with PostgreSQL query planning and migration tooling (Alembic).",
            "Experience writing clean, maintainable pytest test suites with high code coverage."
        ]
    },
    {
        "id": "java_spring_boot_engineer",
        "title": "Staff Java Enterprise Engineer (Spring Boot)",
        "dept": "Enterprise Platform",
        "company": "Apex Global Bank",
        "location": "Charlotte, NC / Remote",
        "overview": "Anchor enterprise financial transaction backends using Java 21, Spring Boot 3, and event-driven architectures.",
        "responsibilities": [
            "Build resilient, high-volume banking microservices using Spring Boot, Spring Cloud, and Java Virtual Threads.",
            "Implement transactional integrity with distributed transaction patterns (Saga pattern) across services.",
            "Manage enterprise relational databases (Oracle, PostgreSQL) with Hibernate / JPA performance tuning.",
            "Enforce strict OAuth2, mTLS, and RBAC security policies across enterprise API gateways."
        ],
        "requirements": [
            "6+ years of enterprise backend engineering in Java (Java 17/21, Spring Boot).",
            "Deep understanding of concurrency, JVM garbage collection tuning (ZGC, G1), and memory leaks.",
            "Demonstrated experience with event streaming using Kafka and schema registries (Avro/Protobuf).",
            "Experience in regulated environments (FinTech, Banking, Healthcare)."
        ]
    },
    {
        "id": "csharp_dotnet_engineer",
        "title": "Senior .NET Core Backend Engineer (C# / Azure)",
        "dept": "Cloud Solutions",
        "company": "CoreTech Enterprise",
        "location": "Atlanta, GA / Remote",
        "overview": "Build scalable cloud services using C#, .NET 8, ASP.NET Core, and Microsoft Azure cloud primitives.",
        "responsibilities": [
            "Design enterprise REST APIs and background services with ASP.NET Core and Entity Framework Core.",
            "Deploy and monitor containerized workloads on Azure Kubernetes Service (AKS) and Azure Container Apps.",
            "Optimize SQL queries and stored procedures on Azure SQL and Cosmos DB.",
            "Automate infrastructure provisioning using Azure Bicep / Terraform and Azure DevOps pipelines."
        ],
        "requirements": [
            "4+ years of experience with C# and .NET Core (version 6/8).",
            "Hands-on expertise with Azure Cloud services (Azure App Services, Key Vault, Service Bus).",
            "Proficiency with relational database modeling, indexing, and EF Core query tuning.",
            "Knowledge of automated unit testing with xUnit, Moq, and FluentAssertions."
        ]
    },
    {
        "id": "api_platform_engineer",
        "title": "API Gateway & Platform Engineer",
        "dept": "Developer Platform",
        "company": "ConnectScale Inc",
        "location": "San Francisco, CA / Remote",
        "overview": "Build the core API gateway and developer portal handling 1 billion monthly API calls across multi-region edge nodes.",
        "responsibilities": [
            "Design and maintain high-scale API gateways using Kong, Envoy, or Traefik.",
            "Implement distributed rate limiting, token bucket algorithms, and caching with Redis clusters.",
            "Develop developer-facing SDKs and OpenAPI documentation automation pipelines.",
            "Monitor gateway latency percentiles (P95, P99) and author automatic circuit breaker rules."
        ],
        "requirements": [
            "4+ years in backend infrastructure and API gateway architecture.",
            "Experience configuring and tuning Envoy proxy, Kong, or Nginx.",
            "Strong proficiency in Go, Python, or Lua scripting for gateway filters.",
            "Solid grasp of HTTP/2, HTTP/3, gRPC, and REST architectural standards."
        ]
    },
    {
        "id": "game_backend_engineer",
        "title": "Multiplayer Game Backend Engineer (C++ / Go)",
        "dept": "Game Infrastructure",
        "company": "Voxel Dynamics Games",
        "location": "Los Angeles, CA / Remote",
        "overview": "Design authoritative game server backends and low-latency matchmaking services for multiplayer games.",
        "responsibilities": [
            "Build authoritative state synchronization and UDP packet handling services using C++ or Go.",
            "Design scalable matchmaking engines supporting ELO/MMR rating algorithms and low latency matching.",
            "Orchestrate dedicated game server fleets dynamically on Kubernetes using Agones.",
            "Implement anti-cheat telemetry validation and anomalous gameplay behavior detection."
        ],
        "requirements": [
            "3+ years of game backend or real-time networking software engineering.",
            "Understanding of UDP, client-side prediction, entity interpolation, and tick rates.",
            "Hands-on experience with Agones, Kubernetes, or AWS GameLift.",
            "Strong passion for multiplayer gaming and competitive fairness."
        ]
    },
    {
        "id": "ruby_rails_architect",
        "title": "Lead Ruby on Rails Engineer",
        "dept": "Product Engineering",
        "company": "BrightSpark SaaS",
        "location": "Portland, OR / Remote",
        "overview": "Modernize and scale high-revenue B2B SaaS web applications utilizing Ruby on Rails 7+, Hotwire, and PostgreSQL.",
        "responsibilities": [
            "Architect and optimize large-scale Ruby on Rails 7 codebases powering enterprise SaaS workflows.",
            "Eliminate N+1 query bottlenecks and tune ActiveRecord queries on multi-terabyte PostgreSQL databases.",
            "Implement responsive real-time client interfaces using Turbo, Stimulus, and ActionCable.",
            "Manage background job scheduling and failure retries with Sidekiq Pro and Redis."
        ],
        "requirements": [
            "5+ years of dedicated professional Ruby on Rails software development.",
            "Deep understanding of Rails internal architectures, middleware, and ActiveRecord optimizations.",
            "Expertise with PostgreSQL query performance analysis and Redis caching.",
            "History of improving test suite velocity with RSpec."
        ]
    },
    {
        "id": "elixir_scalability_engineer",
        "title": "Distributed Systems Engineer (Elixir / Erlang OTP)",
        "dept": "Real-Time Systems",
        "company": "ChatSphere Networks",
        "location": "Remote",
        "overview": "Build massively concurrent, fault-tolerant messaging backends using Elixir, Phoenix Channels, and Erlang BEAM.",
        "responsibilities": [
            "Design highly available real-time communication systems servicing millions of persistent WebSockets.",
            "Leverage Erlang OTP supervision trees, GenServer, and ETS tables for resilient in-memory caching.",
            "Optimize cluster discovery and distributed pub/sub across multi-region server nodes.",
            "Debug live BEAM nodes using Observer, Telemetry, and Erlang diagnostics."
        ],
        "requirements": [
            "3+ years of production experience in Elixir or Erlang.",
            "Deep comprehension of the BEAM virtual machine, actor model, and supervision strategies.",
            "Experience building scalable real-time systems with Phoenix Framework.",
            "Strong grounding in distributed consensus and eventual consistency."
        ]
    },
    {
        "id": "graphics_cplusplus_engineer",
        "title": "3D Graphics & Rendering Engineer (Vulkan / Metal)",
        "dept": "Rendering Core",
        "company": "OmniRender Studios",
        "location": "San Francisco, CA",
        "overview": "Engineer cutting-edge real-time 3D rendering pipelines using modern graphics APIs (Vulkan, DirectX 12, Metal).",
        "responsibilities": [
            "Author low-level GPU rendering code and compute shaders in HLSL / GLSL.",
            "Implement physically based rendering (PBR), real-time ray tracing, and spatial partitioning algorithms.",
            "Profile GPU execution times, memory barriers, and draw calls using RenderDoc and NVIDIA Nsight.",
            "Collaborate with 3D artists to optimize asset pipelines and shader complexity."
        ],
        "requirements": [
            "4+ years of C++ software engineering with modern graphics APIs (Vulkan, DX12, or Metal).",
            "Strong mathematical foundation in linear algebra, 3D transformations, and computational geometry.",
            "Experience writing and profiling compute and fragment shaders.",
            "Degree in Computer Science, Math, or equivalent industry graphics track record."
        ]
    },
    {
        "id": "web3_solidity_engineer",
        "title": "Smart Contract Security & Protocol Engineer",
        "dept": "Blockchain Core",
        "company": "Nexus Protocol",
        "location": "Remote",
        "overview": "Design, verify, and deploy gas-optimized smart contracts and decentralized protocols on EVM chains.",
        "responsibilities": [
            "Write secure, gas-efficient Solidity smart contracts adhering to ERC standards (ERC-20, ERC-721, ERC-4626).",
            "Perform formal verification and exhaustive fuzz testing using Foundry (Forge) and Echidna.",
            "Audit contracts against reentrancy, integer overflow, front-running, and flash loan attacks.",
            "Coordinate third-party security audits and deploy contracts to testnets and mainnet."
        ],
        "requirements": [
            "3+ years of professional Solidity development with deployed mainnet protocols.",
            "Deep understanding of the EVM memory model, opcodes, and gas optimization strategies.",
            "Mastery of modern smart contract testing with Foundry.",
            "Uncompromising focus on cryptographic security and defensive programming."
        ]
    },
    {
        "id": "microservices_architect",
        "title": "Principal Microservices Solutions Architect",
        "dept": "Architecture Office",
        "company": "GlobalScale Enterprises",
        "location": "Dallas, TX / Remote",
        "overview": "Guide engineering teams across domain-driven design, service decomposition, and enterprise microservice standards.",
        "responsibilities": [
            "Define enterprise architecture blueprints for decoupling legacy monoliths into bounded microservices.",
            "Establish standards for API versioning, event schemas, idempotency, and asynchronous workflows.",
            "Lead architectural review boards (ARBs) and author Architecture Decision Records (ADRs).",
            "Ensure system resilience across network failures using retry budgets, timeouts, and fallback patterns."
        ],
        "requirements": [
            "8+ years in software engineering with 3+ years in principal or architecture leadership roles.",
            "Proven track record executing successful monolith-to-microservice decomposition.",
            "Mastery of Domain-Driven Design (DDD) and distributed systems transaction patterns.",
            "Exceptional written and oral communication skills for executive alignment."
        ]
    },
    {
        "id": "desktop_csharp_engineer",
        "title": "Senior Desktop Application Developer (WPF / .NET)",
        "dept": "Client Software",
        "company": "ProDesk Systems",
        "location": "Minneapolis, MN / Remote",
        "overview": "Develop responsive, high-fidelity desktop applications for financial analysts using C#, WPF, and MVVM.",
        "responsibilities": [
            "Design and build performant desktop UI controls in XAML with advanced data virtualization.",
            "Handle low-latency real-time market data streaming over WebSockets without freezing the UI thread.",
            "Implement multi-threaded data caching and local IPC communication.",
            "Manage MSIX packaging, auto-update mechanisms, and code signing certificates."
        ],
        "requirements": [
            "4+ years developing native Windows desktop software with C# and WPF / .NET Core.",
            "Deep understanding of XAML styling, templates, and the MVVM architecture pattern.",
            "Strong skills in multithreaded UI programming (Dispatcher, async/await, Task Parallel Library).",
            "Experience building high-data-volume, real-time desktop interfaces."
        ]
    },
    {
        "id": "automation_tools_engineer",
        "title": "Developer Tooling & CLI Engineer (Go / Python)",
        "dept": "Developer Productivity",
        "company": "ToolCraft Solutions",
        "location": "Remote",
        "overview": "Build internal command-line utilities, local dev environment orchestrators, and automated developer tools.",
        "responsibilities": [
            "Develop cross-platform CLI tools in Go using Cobra and Bubbletea for 2,000+ internal software engineers.",
            "Automate local development environment setup using Docker Compose, DevContainers, and Nix.",
            "Track and optimize developer build velocities, test execution durations, and onboarding friction.",
            "Collect developer feedback and iterate on CLI ergonomics, documentation, and shell completions."
        ],
        "requirements": [
            "3+ years building developer tools, CLI utilities, or developer productivity software in Go or Python.",
            "Familiarity with containerization (Docker), shell scripting (Bash, Zsh), and package managers.",
            "Passion for developer experience (DevEx) and ergonomic API/CLI design.",
            "Strong empathy for engineering workflows."
        ]
    },
    {
        "id": "api_integration_specialist",
        "title": "Senior Enterprise Integration Engineer",
        "dept": "Customer Engineering",
        "company": "Synapse Connect",
        "location": "Chicago, IL / Remote",
        "overview": "Build custom enterprise connectors, ETL webhooks, and third-party data synchronization pipelines.",
        "responsibilities": [
            "Build robust integration middleware connecting customer CRMs (Salesforce, HubSpot) and ERPs (NetSuite).",
            "Implement resilient webhook ingestion endpoints with automated retry buffers and HMAC verification.",
            "Map complex disparate JSON/XML data payloads into canonical internal schema representations.",
            "Troubleshoot client authentication issues across OAuth 2.0, SAML, and API token handshakes."
        ],
        "requirements": [
            "4+ years building external API integrations and data synchronization middleware.",
            "Experience integrating with major SaaS platforms (Salesforce, Stripe, Shopify, Workday).",
            "Strong coding skills in Python, Node.js, or Java.",
            "Patience and problem-solving skills for debugging third-party API edge cases."
        ]
    }
]

# Additional 85 roles generated programmatically across all remaining domains
# We will construct high-density roles covering ML, Data, DevOps, Security, QA, and Verticals.

CATEGORIES_EXTENDED = [
    # --- Category 2: Machine Learning & AI (15 Roles) ---
    ("senior_ai_platform_engineer", "Senior AI Platform Engineer (vLLM / Triton)", "AI Infrastructure", "Horizon Intelligence",
     "Architect multi-tenant GPU serving clusters, KV-cache quantization, and production RAG pipelines.",
     ["Deploy vLLM and TensorRT-LLM on Kubernetes.", "Optimize continuous batching and token latency.", "Build semantic caching layers.", "Manage GPU cluster telemetry."],
     ["4+ years ML systems experience.", "PyTorch, CUDA, vLLM, TensorRT-LLM.", "Kubernetes GPU orchestration.", "Vector database integration (Milvus/Qdrant)."]),
    
    ("computer_vision_engineer", "Computer Vision & Edge ML Engineer", "Perception Systems", "Visionary AI",
     "Train and optimize real-time object detection and segmentation models for edge deployment.",
     ["Train YOLO and Vision Transformers using PyTorch.", "Quantize models to INT8 using TensorRT and ONNX Runtime.", "Deploy inference models to NVIDIA Jetson devices.", "Benchmark FPS and latency bottlenecks."],
     ["3+ years computer vision experience.", "PyTorch, OpenCV, TensorRT, ONNX.", "Edge deployment on ARM/Jetson.", "Deep knowledge of CNNs and Vision Transformers."]),

    ("nlp_research_scientist", "NLP Research Scientist (LLM Alignment)", "AI Research", "DeepLexicon Labs",
     "Research hallucination mitigation, RLHF, and DPO fine-tuning methodologies for enterprise language models.",
     ["Develop preference optimization algorithms (DPO, KTO) for model alignment.", "Evaluate hallucination rates across open domain benchmark datasets.", "Author scientific papers and submit to NeurIPS / ACL / EMNLP.", "Design synthetic data filtering pipelines."],
     ["PhD or MS in NLP, Machine Learning, or Computational Linguistics.", "Track record of publications in top NLP conferences.", "Hands-on experience fine-tuning 7B-70B parameter models.", "Deep PyTorch expertise."]),

    ("mlops_platform_engineer", "Staff MLOps Platform Engineer", "ML Infrastructure", "ModelScale Tech",
     "Build automated model registry, feature store, and continuous model training/deployment pipelines.",
     ["Orchestrate automated training DAGs with Kubeflow Pipelines and Ray.", "Maintain enterprise feature store with Feast and Redis.", "Implement automated drift detection and model retraining triggers using Evidently AI.", "Standardize model packaging with MLflow and Docker."],
     ["5+ years building MLOps infrastructure.", "Kubernetes, Kubeflow, MLflow, Ray.", "Experience with feature stores (Feast, Hopsworks).", "Python and Go proficiency."]),

    ("generative_ai_solutions_architect", "Generative AI Solutions Architect", "Enterprise AI Solutions", "Apex AI Partners",
     "Design custom enterprise Generative AI architectures, RAG pipelines, and agentic workflows for Fortune 500 clients.",
     ["Architect enterprise RAG pipelines with hybrid retrieval, chunking, and re-ranking.", "Evaluate multi-agent orchestration frameworks (LangGraph, CrewAI).", "Address enterprise security, guardrails (NeMo Guardrails), and data isolation.", "Lead technical executive workshops and client design reviews."],
     ["5+ years client-facing technical architecture experience.", "Deep practical knowledge of LLM application patterns and RAG.", "Familiarity with cloud platforms (AWS Bedrock, Azure OpenAI, GCP Vertex).", "Outstanding presentation skills."]),

    ("llm_evaluation_engineer", "LLM Evaluation & Benchmarking Engineer", "Model Reliability", "TrustAI Corp",
     "Develop rigorous automated evaluation harnesses to benchmark LLM accuracy, safety, and retrieval precision.",
     ["Build programmatic evaluation suites measuring Ragas metrics (Context Precision, Faithfulness, Answer Relevance).", "Curate high-quality ground-truth evaluation datasets with human and LLM-as-a-judge pipelines.", "Automate adversarial red-teaming tests for prompt injection and jailbreaks.", "Generate standardized model cards and benchmark telemetry reports."],
     ["3+ years in NLP/ML evaluation or data science.", "Experience with evaluation frameworks (Ragas, TruLens, DeepEval).", "Strong Python programming and statistical testing skills.", "Familiarity with prompt engineering and LLM failure modes."]),

    ("speech_audio_ml_engineer", "Speech & Audio ML Engineer (ASR / TTS)", "Audio AI", "SoundFlow Tech",
     "Train and deploy low-latency automatic speech recognition (ASR) and text-to-speech (TTS) neural models.",
     ["Fine-tune Whisper and Conformer architectures for domain-specific vocabulary.", "Optimize neural vocoder inference (HiFi-GAN) for real-time conversational streaming.", "Build acoustic noise reduction and voice activity detection (VAD) algorithms.", "Deploy audio models using C++ and ONNX Runtime."],
     ["3+ years in speech processing or audio machine learning.", "Deep knowledge of acoustic feature extraction (MFCCs, spectrograms).", "PyTorch, HuggingFace, Kaldi or NeMo.", "Experience with streaming audio protocols (WebRTC)."]),

    ("autonomous_driving_perception_engineer", "Autonomous Vehicle Perception Engineer", "Autonomous Mobility", "DriveCore Labs",
     "Develop 3D LiDAR point cloud and multi-camera sensor fusion pipelines for autonomous vehicles.",
     ["Train 3D object detection models on point clouds using PointNet and BEVFormer.", "Implement Kalman filters for multi-sensor tracking and state estimation.", "Optimize perception algorithms to meet deterministic safety-critical latency budgets (<30ms).", "Validate algorithms against simulation suites and real-world drive data."],
     ["3+ years perception engineering for robotics or autonomous driving.", "C++17 and Python proficiency.", "Experience with LiDAR, radar, and camera sensor fusion.", "Knowledge of ROS / ROS2 and automotive safety standards (ISO 26262)."]),

    ("recommendation_systems_engineer", "Senior Recommendation Systems Engineer", "Discovery & Personalization", "ShopStream Media",
     "Architect multi-stage personalized recommendation engines handling 50M active shoppers.",
     ["Design candidate generation (two-tower models) and re-ranking deep neural networks.", "Optimize negative sampling and cold-start candidate discovery.", "Deploy real-time inference with Triton Inference Server and Milvus vector retrieval.", "Run online A/B testing measuring click-through rate (CTR) and conversion."],
     ["4+ years building large-scale recommendation or search systems.", "Deep learning for recsys (Two-Tower, DLRM, Transformer-based).", "Distributed training with Ray or PyTorch DDP.", "Solid understanding of online experimentation and A/B testing statistics."]),

    ("reinforcement_learning_engineer", "Reinforcement Learning Engineer", "Decision Intelligence", "OptimalPath AI",
     "Design reinforcement learning algorithms for robotics manipulation and dynamic resource allocation.",
     ["Implement deep RL algorithms (PPO, SAC, TD3) in simulation environments (Isaac Gym, MuJoCo).", "Design reward shaping functions and domain randomization techniques for sim-to-real transfer.", "Optimize policy network inference on onboard embedded hardware.", "Author technical documentation and benchmark results."],
     ["3+ years experience with reinforcement learning and optimal control.", "PyTorch, Stable-Baselines3, or RLlib.", "Experience with physics simulators (MuJoCo, Isaac Gym).", "Strong mathematical foundation in MDPs and stochastic calculus."]),

    ("ml_data_pipeline_engineer", "Machine Learning Data Pipeline Engineer", "Data Operations", "DataNexus AI",
     "Build scalable data ingestion, synthetic generation, and multimodal annotation pipelines for AI training.",
     ["Design petabyte-scale data preprocessing pipelines using Apache Spark and Ray Data.", "Implement automated image/video quality filtering and deduplication using perceptual hashing.", "Manage human-in-the-loop data labeling pipelines with Label Studio.", "Track training dataset lineage and versioning with DVC."],
     ["3+ years in data engineering for machine learning.", "Apache Spark, Ray Data, Python.", "Experience with dataset versioning tools (DVC, LakeFS).", "Understanding of multimodal data formats (Parquet, WebDataset, TFRecord)."]),

    ("edge_ai_optimization_engineer", "Edge AI Optimization & Compilers Engineer", "Edge Intelligence", "TinySilicon Corp",
     "Optimize deep learning models for custom NPUs, microcontrollers, and low-power hardware.",
     ["Utilize TVM and MLIR to compile deep neural networks for proprietary NPU architectures.", "Implement pruning, quantization (INT4, FP8), and knowledge distillation techniques.", "Profile memory bandwidth and compute utilization on target embedded hardware.", "Collaborate with hardware architects to specify next-generation accelerator instructions."],
     ["3+ years in deep learning compiler engineering or edge optimization.", "Knowledge of TVM, MLIR, ONNX, TensorRT.", "C++ and Python mastery.", "Understanding of computer architecture and cache hierarchies."]),

    ("quantitative_ml_researcher", "Quantitative ML Researcher", "Algorithmic Trading", "AlphaMetrics Capital",
     "Develop predictive statistical and machine learning alpha signals for high-frequency financial markets.",
     ["Research non-linear predictive signals using gradient boosting and temporal graph neural networks.", "Clean and normalize tick-level limit order book (LOB) financial data.", "Backtest trading strategies accounting for market impact, slippage, and execution latency.", "Collaborate with quant developers to deploy models into ultra-low-latency execution engines."],
     ["MS or PhD in Mathematics, Statistics, Physics, or Computer Science.", "Proficiency in Python (NumPy, Pandas, Scipy, Polars) and C++.", "Deep understanding of time-series analysis and financial market microstructure.", "Experience working with limit order book data."]),

    ("prompt_and_finetuning_engineer", "LLM Prompt & Fine-Tuning Engineer", "Applied AI", "PromptFlow Tech",
     "Specialize in parameter-efficient fine-tuning (PEFT/LoRA) and deterministic prompt orchestration.",
     ["Fine-tune open-source models (Llama 3, Mistral) on domain-specific corpora using QLoRA and Unsloth.", "Author complex prompt chains with structured JSON outputs and function calling.", "Evaluate prompt robustness against edge cases and system prompt leakage.", "Optimize prompt token efficiency to reduce cloud API expenditures."],
     ["2+ years working with LLMs, prompt engineering, and fine-tuning.", "Hands-on experience with HuggingFace, PEFT, TRL, and Unsloth.", "Familiarity with structured output libraries (Instructor, Outlines).", "Attention to detail and linguistic nuance."]),

    ("ai_safety_and_alignment_auditor", "AI Safety & Alignment Auditor", "Governance & Trust", "EthicAI Labs",
     "Perform safety red-teaming, bias audits, and regulatory compliance assessments on generative models.",
     ["Design adversarial prompt injection suites to test model safety guardrails.", "Measure demographic bias and representational parity in generative outputs.", "Author comprehensive AI safety governance reports compliant with EU AI Act standards.", "Collaborate with legal and engineering teams to establish acceptable use policies."],
     ["3+ years in AI ethics, safety, red-teaming, or model auditing.", "Familiarity with regulatory frameworks (EU AI Act, NIST AI RMF).", "Technical ability to script automated test suites in Python.", "Strong technical writing and cross-functional communication."]),

    # --- Category 3: Data Engineering & Analytics (12 Roles) ---
    ("staff_data_platform_engineer", "Staff Data Platform Engineer (Spark / Lakehouse)", "Data Platform", "Horizon Intelligence",
     "Anchor streaming lakehouse architectures using Apache Spark, Apache Iceberg, and Delta Lake.",
     ["Migrate Hadoop workloads to Databricks / Iceberg.", "Build streaming pipelines with Kafka and Flink.", "Enforce data quality with Great Expectations and dbt.", "Optimize cloud warehouse spend."],
     ["6+ years data engineering.", "Apache Spark, Delta Lake, Apache Iceberg, Kafka.", "AWS or GCP data ecosystems.", "Mentorship and architectural leadership."]),

    ("streaming_data_engineer", "Real-Time Streaming Data Engineer (Flink / Kafka)", "Streaming Infrastructure", "StreamWave Analytics",
     "Build sub-second event streaming platforms processing 500,000 events/second with Apache Flink and Kafka.",
     ["Develop stateful streaming applications using Apache Flink and Kafka Streams.", "Implement exactly-once processing semantics across distributed consumer workers.", "Manage Kafka broker clusters, partition rebalancing, and Schema Registry.", "Monitor streaming consumer lag and author automated backpressure handlers."],
     ["4+ years building production streaming pipelines.", "Apache Flink, Apache Kafka, Scala or Java.", "Deep understanding of event-time processing and watermarking.", "Experience with distributed storage (ClickHouse, Pinot, RocksDB)."]),

    ("analytics_engineer_dbt", "Senior Analytics Engineer (dbt & Snowflake)", "Data Analytics", "ClearMetrics",
     "Transform raw data lakehouse tables into clean, tested, documented dimensional models for business intelligence.",
     ["Design modular dbt models following Kimball dimensional modeling best practices.", "Implement automated data quality test suites using dbt tests and Great Expectations.", "Optimize Snowflake query performance, clustering keys, and virtual warehouse sizing.", "Collaborate with business stakeholders to define canonical business KPI metrics."],
     ["4+ years in analytics engineering or BI development.", "Mastery of SQL, dbt (Core or Cloud), and Snowflake / BigQuery.", "Strong dimensional modeling fundamentals (star schema, facts, dimensions).", "Familiarity with Git version control and CI/CD for analytics."]),

    ("big_data_architect_spark", "Principal Big Data Architect (Databricks / Spark)", "Enterprise Data", "OmniScale Data",
     "Design petabyte-scale analytical data platforms and govern architectural standards across multiple data teams.",
     ["Architect enterprise Databricks lakehouse platforms on AWS utilizing Unity Catalog.", "Optimize Spark query physical execution plans, broadcast joins, and shuffle partitions.", "Establish enterprise data governance, lineage tracking, and compliance frameworks.", "Evaluate emerging storage formats (Iceberg vs Delta vs Hudi) and provide architectural roadmaps."],
     ["8+ years data engineering with 3+ years in principal or architecture roles.", "Deep mastery of the Apache Spark engine internals and memory tuning.", "Hands-on experience with Databricks Unity Catalog and cloud infrastructure (Terraform).", "Executive communication skills."]),

    ("bi_visualization_engineer", "Senior BI & Dashboard Engineer (Tableau / Power BI)", "Business Intelligence", "InsightHub Analytics",
     "Design intuitive, responsive analytical dashboards delivering operational insights to company leadership.",
     ["Build interactive, executive-ready dashboards in Tableau and Microsoft Power BI.", "Optimize underlying SQL analytical views and data extraction schedules to ensure fast load times.", "Implement row-level security (RLS) to ensure data confidentiality across departments.", "Conduct user research with business analysts to refine visual UX and dashboard ergonomics."],
     ["4+ years building enterprise dashboards in Tableau or Power BI.", "Advanced SQL writing complex CTEs, window functions, and aggregations.", "Understanding of visual perception, color theory, and chart choice fundamentals.", "Strong stakeholder management skills."]),

    ("data_governance_specialist", "Enterprise Data Governance & Quality Engineer", "Data Governance", "SecureData Corp",
     "Lead enterprise data cataloging, lineage tracking, data privacy, and metadata management across 200+ databases.",
     ["Implement enterprise data catalog platforms (Alation, Collibra, or Apache Atlas).", "Map end-to-end data lineage from source transactional databases to reporting dashboards.", "Enforce automated data quality SLA monitoring and incident alerting.", "Audit compliance with GDPR, CCPA, and industry data retention standards."],
     ["4+ years in data governance, data quality, or data management.", "Experience with enterprise data catalog tools (Collibra, Alation, Unity Catalog).", "Strong SQL and data profiling capabilities.", "Familiarity with regulatory data compliance laws."]),

    ("realtime_ingestion_engineer", "High-Throughput Data Ingestion Engineer", "Core Ingestion", "FastIngest Tech",
     "Build resilient connectors and ingestion daemons moving terabytes of operational telemetry into analytical warehouses.",
     ["Design distributed ingestion workers using Go, Python, and Kafka.", "Handle schema drift and unannounced upstream payload changes gracefully without pipeline failure.", "Implement exponential backoff retry buffers and dead-letter queues (DLQ).", "Maintain 99.99% ingestion availability SLAs."],
     ["3+ years in distributed data ingestion or backend pipelines.", "Proficiency in Go or Python.", "Hands-on experience with Kafka, Debezium, or AWS Kinesis.", "Experience with schema validation (JSON Schema, Avro, Protobuf)."]),

    ("database_administrator_postgres", "Senior Database Administrator (PostgreSQL DBA)", "Database Operations", "DataCore Systems",
     "Manage, scale, tune, and ensure 99.999% availability for enterprise PostgreSQL database clusters.",
     ["Tune PostgreSQL configurations, shared buffers, work_mem, and autovacuum parameters.", "Manage high-availability clusters using Patroni, Raft, and streaming replication.", "Execute zero-downtime major version database upgrades on terabyte-scale databases.", "Analyze slow query logs, pg_stat_statements, and create optimal indexes."],
     ["5+ years dedicated PostgreSQL administration in production environments.", "Deep expertise with replication, backups (pgBackRest), and failover tooling (Patroni).", "Linux systems performance tuning and disk I/O profiling.", "Experience with connection pooling (PgBouncer)."]),

    ("vector_database_systems_engineer", "Vector Database Systems Engineer (Milvus / Qdrant)", "Vector Infrastructure", "VectorScale Corp",
     "Design and scale high-throughput vector index storage and similarity search infrastructure.",
     ["Deploy and manage distributed Milvus, Qdrant, or Pinecone clusters on Kubernetes.", "Tune Approximate Nearest Neighbor (ANN) index algorithms (HNSW, IVF-PQ) for recall vs latency trade-offs.", "Benchmark vector search query latency under 10,000 QPS load.", "Implement hybrid sparse-dense vector storage architectures."],
     ["3+ years in vector search, IR, or high-performance distributed storage.", "Hands-on experience with Milvus, Qdrant, FAISS, or ChromaDB.", "Understanding of distance metrics (Cosine, Euclidean, Dot Product) and vector quantization.", "Kubernetes deployment expertise."]),

    ("lakehouse_optimization_engineer", "Lakehouse Performance & FinOps Engineer", "Cloud FinOps", "CloudSave Data",
     "Audit and optimize cloud data warehouse storage and compute expenses across Snowflake and Databricks.",
     ["Analyze Snowflake query profiling and warehouse utilization to eliminate wasted compute credits.", "Optimize Delta Lake and Iceberg table file compaction (Z-Ordering, bin-packing) to accelerate queries.", "Implement automated warehouse auto-suspend and scaling policies.", "Deliver monthly executive cost attribution and savings reports."],
     ["4+ years in data engineering with deep focus on query optimization and cloud costs.", "Expertise in Snowflake, Databricks, or BigQuery internal cost models.", "Advanced SQL performance tuning and index optimization.", "Demonstrated history of saving $100k+ in cloud compute spend."]),

    ("master_data_management_lead", "Master Data Management (MDM) Architect", "Enterprise Data", "GlobalCorp Enterprise",
     "Design customer and product master data harmonization, deduplication, and entity resolution systems.",
     ["Build deterministic and probabilistic entity resolution pipelines using machine learning.", "Establish golden record creation workflows across disparate CRM and ERP records.", "Implement real-time master data synchronization via REST APIs and Kafka.", "Conduct steering committee reviews with enterprise business units."],
     ["6+ years in master data management (MDM) or enterprise data integration.", "Experience with entity resolution algorithms and fuzzy matching (Levenshtein, Jaro-Winkler).", "Strong database and data modeling expertise.", "Enterprise stakeholder management skills."]),

    ("data_quality_engineer", "Automated Data Quality & Reliability Engineer", "Data Assurance", "ReliableData Labs",
     "Build automated validation frameworks ensuring data pipelines never ship silent corruptions to production.",
     ["Author automated data expectation test suites using Great Expectations and Soda Core.", "Implement automated schema migration checks that block breaking upstream column deletions.", "Build Slack and PagerDuty notification webhooks for data pipeline anomalies.", "Collaborate with data engineers to establish SLOs for data freshness and validity."],
     ["3+ years in data quality, data testing, or analytics engineering.", "Strong Python and SQL scripting skills.", "Experience with Great Expectations, Soda, or Monte Carlo.", "Obsessive attention to data correctness and edge cases."]),

    # --- Category 4: Cloud, DevOps & SRE (10 Roles) ---
    ("senior_site_reliability_engineer", "Lead Site Reliability Engineer (Kubernetes & SRE)", "Site Reliability", "StripeGuard Financial",
     "Design and operate multi-region Kubernetes clusters with automated chaos engineering and 99.995% uptime.",
     ["Manage 14+ production EKS clusters hosting 200+ microservices.", "Establish SLOs/SLIs for core payment pathways.", "Implement declarative GitOps rollouts using ArgoCD.", "Conduct chaos engineering drills with Chaos Mesh."],
     ["6+ years SRE/DevOps experience.", "Certified Kubernetes Administrator (CKA).", "Terraform, ArgoCD, Prometheus, Grafana.", "Incident management and root cause analysis."]),

    ("cloud_infrastructure_architect_aws", "Principal Cloud Solutions Architect (AWS)", "Cloud Architecture", "CloudScale Consulting",
     "Design secure, multi-account, highly available cloud foundations for enterprise migrations.",
     ["Design multi-account AWS Organizations landing zones with Control Tower and IAM Identity Center.", "Architect cross-region disaster recovery and VPC transit gateway network topologies.", "Ensure compliance with AWS Well-Architected Framework across security, reliability, and cost.", "Advise CTOs and VP of Engineering on cloud migration strategies."],
     ["8+ years cloud infrastructure experience with AWS.", "AWS Certified Solutions Architect – Professional.", "Deep expertise in Terraform, AWS Organizations, and networking.", "Executive client advisory experience."]),

    ("multicloud_kubernetes_architect", "Multi-Cloud Kubernetes Platform Architect", "Platform Engineering", "KubeSphere Global",
     "Standardize declarative container platforms across AWS EKS, GCP GKE, and bare-metal environments.",
     ["Build standardized enterprise Kubernetes platforms across AWS, GCP, and on-premise clusters.", "Implement service mesh infrastructure using Istio for mTLS, traffic splitting, and telemetry.", "Standardize cluster admission control using Kyverno and Open Policy Agent (OPA).", "Train internal engineering teams on cloud-native best practices."],
     ["6+ years dedicated Kubernetes platform engineering.", "Deep knowledge of Kubernetes internals, API extensions, and operators.", "Experience with Istio, ArgoCD, and Helm.", "Multi-cloud operational experience."]),

    ("cicd_automation_engineer", "CI/CD Pipeline & Build Release Engineer", "Developer Infrastructure", "FastDeploy Systems",
     "Build ultra-fast, secure automated build, test, and deployment pipelines using GitHub Actions.",
     ["Design reusable GitHub Actions workflows used by 500+ software engineers.", "Implement container image caching, dependency caching, and parallelized test runners.", "Integrate automated security gates (SAST, DAST, container scanning) into merge queues.", "Maintain 99.9% CI runner cluster availability on self-hosted autoscaling runners."],
     ["4+ years building automated CI/CD pipelines.", "Mastery of GitHub Actions, GitLab CI, or Jenkins.", "Docker, Linux shell scripting, and Python.", "Familiarity with container security scanning tools (Trivy, Snyk)."]),

    ("finops_cloud_cost_architect", "Cloud FinOps & Infrastructure Optimization Architect", "FinOps", "CloudBudget AI",
     "Lead organizational cloud cost attribution, spot instance adoption, and commitment discount strategies.",
     ["Implement granular cost allocation tagging and dashboards across multi-account AWS environments.", "Lead spot instance migration strategies for fault-tolerant background workloads.", "Manage AWS Savings Plans and Reserved Instance portfolio optimization.", "Collaborate with engineering leads to establish monthly team cloud budgets."],
     ["4+ years in cloud infrastructure with dedicated focus on FinOps.", "FinOps Certified Practitioner (FOCP) preferred.", "Deep knowledge of AWS Cost Explorer, CUR (Cost and Usage Report), and Datadog.", "Strong analytical and cross-functional leadership skills."]),

    ("observability_telemetry_engineer", "Observability & Telemetry Infrastructure Engineer", "Observability Core", "TracePoint Systems",
     "Build unified distributed tracing, metrics, and log aggregation platforms for high-scale microservices.",
     ["Deploy and manage OpenTelemetry collectors, Grafana Tempo, Prometheus, and VictoriaMetrics.", "Implement standardized distributed trace propagation headers (W3C Trace Context) across services.", "Optimize telemetry ingestion pipelines and author metric drop-rules to curb Datadog billings.", "Create operational dashboards and high-signal alert rules."],
     ["4+ years in observability or SRE roles.", "Experience with OpenTelemetry, Prometheus, Grafana, and ELK Stack.", "Deep understanding of metrics (RED/USE methods), traces, and structured logging.", "Python or Go scripting skills."]),

    ("gitops_automation_specialist", "GitOps Platform Specialist (ArgoCD / Flux)", "Platform Delivery", "DeclarativeCloud",
     "Modernize deployment workflows by enforcing 100% declarative GitOps across enterprise clusters.",
     ["Manage enterprise ArgoCD or Flux deployments syncing thousands of application manifests.", "Implement progressive delivery (canary, blue-green) with Argo Rollouts and Prometheus metrics analysis.", "Secure GitOps repositories with signed commits and automated pull request validation.", "Troubleshoot out-of-sync cluster states and drift reconciliation."],
     ["3+ years in cloud-native deployment with dedicated GitOps tools.", "ArgoCD, Argo Rollouts, Flux, Helm, Kustomize.", "Solid Kubernetes administration foundation.", "Understanding of infrastructure-as-code principles."]),

    ("disaster_recovery_architect", "Disaster Recovery & High-Availability Lead", "Infrastructure Resilience", "ResilienceGuard",
     "Design, test, and automate multi-region failover strategies ensuring zero data loss (RPO=0) during outages.",
     ["Design multi-region active-active and active-passive architecture patterns on AWS.", "Automate database replication and DNS failover using AWS Route 53 and Global Accelerator.", "Orchestrate quarterly game-day disaster recovery simulation drills.", "Author formal disaster recovery runbooks and executive business continuity reports."],
     ["6+ years in high-availability systems and disaster recovery.", "Expertise in distributed database replication (Aurora Global Database, CockroachDB).", "Terraform automation mastery.", "Strong incident leadership and documentation skills."]),

    ("systems_administrator_linux", "Senior Linux Systems Administrator (RHEL / Ubuntu)", "Core Infrastructure", "ServerCore Technologies",
     "Maintain enterprise Linux server fleets, kernel tuning, security hardening, and storage arrays.",
     ["Manage 800+ physical and virtual Linux servers running Red Hat Enterprise Linux and Ubuntu.", "Implement security hardening according to CIS Benchmarks and DISA STIG guidelines.", "Automate server configuration management using Ansible playbooks.", "Diagnose low-level kernel panics, CPU throttling, and network interface bottlenecks."],
     ["5+ years enterprise Linux administration.", "Red Hat Certified Engineer (RHCE) or equivalent.", "Deep knowledge of systemd, SELinux, LVM, and networking (iptables, firewalld).", "Proficiency in Bash and Python scripting."]),

    ("network_infrastructure_engineer", "Cloud & Hybrid Network Infrastructure Engineer", "Network Engineering", "NetBridge Solutions",
     "Design high-performance hybrid network topologies connecting on-premise data centers to AWS via Direct Connect.",
     ["Configure and operate AWS Direct Connect, Transit Gateways, and VPN tunnels.", "Manage BGP routing tables, VPC peering, and overlapping IP address subnetting (NAT gateways).", "Implement cloud network security using AWS Network Firewall and security group automation.", "Troubleshoot packet drops, MTU misconfigurations, and latency anomalies with Wireshark."],
     ["4+ years in enterprise and cloud networking.", "Cisco CCNP or AWS Certified Advanced Networking – Specialty.", "Deep understanding of BGP, IPsec, TCP/IP, and DNS architectures.", "Experience with network automation tools."]),

    # --- Category 5: Cybersecurity & DevSecOps (10 Roles) ---
    ("senior_application_security_engineer", "Senior AppSec & DevSecOps Engineer", "Application Security", "Apex Financial Technologies",
     "Lead automated CI/CD vulnerability scanning, API penetration testing, and zero-trust architectures.",
     ["Operationalize SAST/DAST security scanning with Semgrep and Snyk.", "Perform API penetration testing identifying OWASP Top 10 vulnerabilities.", "Implement Zero-Trust network access with Okta and mTLS.", "Attain and maintain SOC 2 Type II compliance."],
     ["5+ years application security experience.", "OSCP, CISSP, or AWS Security Specialty.", "Semgrep, Snyk, Burp Suite Pro, Falco.", "Proven vulnerability remediation leadership."]),

    ("cloud_security_architect", "Lead Cloud Security Architect (AWS / GCP)", "Information Security", "ShieldCloud Security",
     "Architect cloud security postures, IAM least privilege policies, and automated threat detection.",
     ["Design AWS IAM least-privilege permission boundaries and SCP guardrails.", "Deploy automated cloud security posture management (CSPM) using AWS Security Hub and Wiz.", "Automate threat detection and incident response workflows with AWS GuardDuty and Lambda.", "Conduct cloud architecture security reviews for newly planned enterprise workloads."],
     ["6+ years in cloud security architecture.", "AWS Certified Security – Specialty or CISSP.", "Hands-on experience with CSPM tools (Wiz, Orca, Prisma Cloud).", "Deep knowledge of IAM policies and cryptographic KMS key management."]),

    ("penetration_tester_red_team", "Senior Penetration Tester & Red Teamer", "Offensive Security", "CyberBreach Labs",
     "Conduct black-box penetration tests, red-team adversary simulations, and exploit research.",
     ["Perform external and internal penetration tests across cloud networks and public APIs.", "Simulate real-world adversary tactics based on the MITRE ATT&CK framework.", "Author comprehensive technical vulnerability assessment reports with remediation guidance.", "Develop custom exploitation scripts in Python and Go."],
     ["4+ years in offensive security and ethical hacking.", "Offensive Security Certified Professional (OSCP) or OSCE.", "Proficiency with Burp Suite Pro, Metasploit, Nmap, and Cobalt Strike.", "Deep understanding of web vulnerabilities and privilege escalation."]),

    ("soc_incident_responder", "Senior SOC Incident Response Analyst", "Security Operations", "ThreatShield Defense",
     "Lead real-time threat investigation, malware analysis, and security incident containment.",
     ["Investigate high-severity security alerts across EDR systems (CrowdStrike Falcon, SentinelOne).", "Perform forensic disk, memory, and network packet analysis during suspected breaches.", "Develop automated security orchestration playbooks (SOAR) to isolate compromised endpoints.", "Author formal post-incident root-cause investigation reports."],
     ["3+ years in security operations or incident response.", "GIAC Certified Incident Handler (GCIH) or equivalent.", "Experience with SIEM (Splunk, Microsoft Sentinel) and EDR platforms.", "Strong understanding of Windows and Linux forensic artifacts."]),

    ("identity_access_management_engineer", "Identity & Access Management (IAM) Specialist", "Identity Security", "IdentityLock Corp",
     "Design centralized identity management, Single Sign-On (SSO), and lifecycle access policies.",
     ["Architect enterprise SSO, MFA, and user provisioning using Okta, Azure AD, and SCIM protocol.", "Implement fine-grained Role-Based Access Control (RBAC) and Attribute-Based Access Control (ABAC).", "Automate privileged access management (PAM) and credential rotation with CyberArk.", "Conduct quarterly user access reviews and audit access certifications for compliance."],
     ["4+ years in IAM and identity security.", "Deep understanding of OAuth 2.0, OpenID Connect (OIDC), and SAML 2.0.", "Experience with Okta, Azure AD, or Ping Identity.", "Python scripting for automated identity reconciliation."]),

    ("zero_trust_network_architect", "Zero-Trust Architecture Lead", "Enterprise Security", "TrustZero Systems",
     "Lead company-wide migration from perimeter VPNs to modern software-defined Zero-Trust perimeter networks.",
     ["Deploy secure application access proxies using Cloudflare Access or Zscaler Private Access.", "Enforce mutual TLS (mTLS) with device certificate posture verification across corporate endpoints.", "Collaborate with IT to decommission legacy corporate VPN concentrators.", "Conduct user training and ensure frictionless access for global remote employees."],
     ["5+ years in network security and identity architecture.", "Deep knowledge of zero-trust network principles (NIST SP 800-207).", "Experience with Zscaler, Cloudflare Zero Trust, or Palo Alto Prisma.", "Strong cross-departmental project management skills."]),

    ("vulnerability_management_lead", "Vulnerability Assessment & Remediation Lead", "Security Assurance", "VulnerabilityGuard",
     "Orchestrate vulnerability prioritization, patch tracking, and SLA enforcement across 50 engineering teams.",
     ["Scan enterprise codebases and container registries continuously with Snyk, Tenable, and Trivy.", "Prioritize vulnerability remediation based on EPSS (Exploit Prediction Scoring System) and CVSS scores.", "Collaborate with product teams to track remediation tickets in Jira within strict compliance SLAs.", "Produce monthly executive vulnerability metrics and MTTR reports."],
     ["3+ years in vulnerability management or DevSecOps.", "Familiarity with vulnerability scoring (CVSS, EPSS) and CISA KEV catalog.", "Scripting skills in Python to automate Jira ticket creation.", "Excellent diplomacy and engineering negotiation skills."]),

    ("cryptography_security_engineer", "Applied Cryptography & PKI Engineer", "Cryptographic Security", "CryptoShield Core",
     "Implement cryptographic key lifecycle management, TLS certificate authorities, and data-at-rest encryption.",
     ["Manage enterprise Public Key Infrastructure (PKI) and certificate automation with HashiCorp Vault.", "Advise software engineering teams on secure implementation of symmetric/asymmetric encryption algorithms.", "Conduct architectural planning for Post-Quantum Cryptography (PQC) algorithm migrations.", "Audit cryptographic libraries and random number generation implementations."],
     ["4+ years in applied cryptography, PKI, or security engineering.", "Strong mathematical foundation in number theory and modern cryptography (AES, RSA, ECC).", "Hands-on experience with HashiCorp Vault, AWS KMS, or HSM hardware.", "Programming proficiency in C, C++, or Go."]),

    ("compliance_soc2_auditor", "Technical Security Compliance Specialist (SOC 2 / ISO 27001)", "Security Compliance", "AuditReady Tech",
     "Lead technical evidence collection, continuous compliance automation, and external security audits.",
     ["Automate evidence collection using compliance platforms (Vanta, Drata) integrated with AWS and GitHub.", "Coordinate annual SOC 2 Type II, ISO 27001, and HIPAA third-party independent audits.", "Conduct internal security risk assessments and maintain the corporate Risk Register.", "Train new employees on security awareness and acceptable use policies."],
     ["3+ years in security compliance or IT audit.", "CISA, CISSP, or ISO 27001 Lead Auditor certification preferred.", "Experience with automated compliance software (Vanta, Drata).", "Thorough, organized, and detail-oriented work ethic."]),

    ("devsecops_integration_engineer", "DevSecOps Integration Specialist", "Security Engineering", "PipelineShield",
     "Embed automated security testing seamlessly into developer git commit workflows and pull requests.",
     ["Build pre-commit git hooks and GitHub Actions workflows scanning for hardcoded secrets (TruffleHog).", "Integrate Open Policy Agent (OPA) / Conftest to block insecure Terraform manifests prior to apply.", "Deliver security tool feedback directly into developer PR comments to avoid developer friction.", "Track developer fix rates and security debt across repositories."],
     ["3+ years in DevSecOps or platform engineering.", "Strong proficiency with GitHub Actions, Terraform, and Docker.", "Hands-on experience with secret scanning and policy-as-code tools (OPA, Checkov).", "Empathy for developer velocity."]),

    # --- Category 6: Product, QA & Architecture (8 Roles) ---
    ("lead_sdet_automation", "Lead Software Development Engineer in Test (SDET)", "Quality Engineering", "QualityScale Labs",
     "Design automated end-to-end testing frameworks, parallel execution runners, and performance benchmarks.",
     ["Build scalable automated testing frameworks from scratch using TypeScript and Playwright.", "Integrate automated API integration test suites into GitHub Actions CI pipelines.", "Establish synthetic monitoring health checks running against production endpoints.", "Mentor manual QA testers in test automation and modern test design."],
     ["5+ years as an SDET or test automation engineer.", "Mastery of Playwright, Cypress, or Selenium with TypeScript/Python.", "Strong background in REST and GraphQL API testing.", "Experience with CI/CD integration and test reporting."]),

    ("performance_load_test_engineer", "Senior Performance & Load Testing Engineer (k6 / Locust)", "Reliability Testing", "StressTest Solutions",
     "Design realistic load tests, stress tests, and spike tests uncovering system bottlenecks before major sales events.",
     ["Script complex realistic user simulation scenarios using k6, Locust, or Gatling.", "Execute stress tests scaling to 100,000 concurrent virtual users against staging clusters.", "Analyze database connection pool exhaustion, CPU saturation, and memory limits during load.", "Author executive capacity planning reports with recommendations on autoscaling thresholds."],
     ["4+ years dedicated performance and load testing.", "Proficiency in JavaScript (for k6) or Python (for Locust).", "Deep understanding of HTTP performance, connection reuse, and network latency.", "Experience analyzing distributed trace bottlenecks in Datadog or Grafana."]),

    ("technical_product_manager_ai", "Technical Product Manager (Developer AI Platform)", "Product Management", "AIPlatform Pro",
     "Define the product roadmap, user stories, and developer experience for our enterprise LLM API gateway.",
     ["Conduct user discovery interviews with developer customers to identify developer friction.", "Write detailed Technical Product Requirement Documents (PRDs) for model serving and RAG capabilities.", "Define product telemetry KPIs: developer activation rate, API latency satisfaction, and retention.", "Collaborate closely with engineering leads to prioritize the bi-weekly sprint backlog."],
     ["4+ years as a Technical Product Manager for developer tools or cloud platforms.", "Computer Science background or ability to write API queries and understand architecture diagrams.", "Strong customer empathy and data-driven prioritization.", "Excellent communication and sprint leadership skills."]),

    ("enterprise_solutions_architect", "Senior Enterprise Solutions Architect", "Solutions Engineering", "ScaleBridge Solutions",
     "Partner with enterprise sales teams to architect technical solutions and guide customer proof-of-concepts (POCs).",
     ["Lead technical discovery calls and map customer architectural requirements to product capabilities.", "Build and present custom technical Proof-of-Concept (POC) demonstrations for enterprise clients.", "Address technical security, compliance, and integration objections from customer architects.", "Author whitepapers and reference architectures for common integration patterns."],
     ["5+ years in pre-sales solutions engineering or enterprise architecture.", "Broad understanding of cloud infrastructure, APIs, and security protocols.", "Exceptional presentation and demonstration skills.", "Ability to explain complex technical concepts to non-technical executives."]),

    ("developer_advocate_ai", "Developer Advocate & Technical Evangelist", "Developer Relations", "OpenAI Tooling",
     "Inspire and educate global software developers through open-source sample apps, technical blogs, and conferences.",
     ["Build open-source demonstration applications showcasing modern RAG and generative AI patterns.", "Author deep-dive technical tutorials, blog posts, and comprehensive documentation.", "Speak at major technical conferences (PyCon, KubeCon, AWS re:Invent) and host developer workshops.", "Gather community feedback and advocate internally for developer product improvements."],
     ["3+ years as a software engineer or developer advocate.", "Strong public speaking and engaging technical writing skills.", "Active presence on GitHub, YouTube, or developer communities.", "Passion for teaching and community building."]),

    ("technical_scrum_master", "Agile Coach & Technical Scrum Master", "Agile Office", "AgileFlow Consulting",
     "Facilitate agile ceremonies, remove engineering blockers, and optimize team delivery velocity across 4 squads.",
     ["Facilitate sprint planning, daily standups, backlog refinement, and insightful sprint retrospectives.", "Identify and eliminate cross-team dependencies and engineering blockers.", "Track team metrics: sprint velocity, burndown, cycle time, and pull request review turnaround.", "Foster a blameless culture of continuous engineering improvement and psychological safety."],
     ["4+ years as a Scrum Master or Agile Coach in technical software engineering environments.", "Certified Scrum Master (CSM) or PMI-ACP certification.", "Deep knowledge of Jira workflows, automation, and agile reporting.", "High emotional intelligence and servant leadership mindset."]),

    ("technical_writer_api", "Senior Technical Writer & Documentation Specialist", "Developer Experience", "DocCraft Systems",
     "Create world-class developer documentation, API reference guides, and interactive code walkthroughs.",
     ["Author comprehensive developer quickstart guides, API references, and architecture overviews.", "Maintain documentation-as-code workflows using Markdown, MDX, and Git version control.", "Test all code samples and API curl requests to guarantee 100% accuracy.", "Collaborate with product and engineering teams to document newly shipped features ahead of releases."],
     ["3+ years technical writing for developer-focused software products.", "Familiarity with OpenAPI/Swagger specifications and docs-as-code tools (Docusaurus, Mintlify).", "Basic coding ability in Python or JavaScript to verify sample code.", "Meticulous eye for grammar, clarity, and developer usability."]),

    ("release_engineering_manager", "Software Release Engineering Manager", "Release Operations", "DeployGuard Inc",
     "Manage production release schedules, zero-downtime deployment gates, and hotfix rollback procedures.",
     ["Coordinate weekly production releases across 12 distributed engineering squads.", "Enforce deployment verification gates, staging smoke tests, and rollback runbooks.", "Chair the Change Advisory Board (CAB) to assess release risks and cross-service dependencies.", "Manage emergency hotfix deployments and conduct post-incident release retrospectives."],
     ["5+ years in release management, DevOps, or software engineering leadership.", "Deep understanding of continuous deployment strategies (canary, blue-green, feature flags).", "Strong risk assessment and crisis leadership capabilities.", "Experience with feature flagging platforms (LaunchDarkly)."]),

    # --- Category 7: Industry-Specific Verticals (20 Roles) ---
    ("fintech_payment_architect", "FinTech Payment Gateway Architect", "Core Payments", "PayStream Global",
     "Architect PCI-DSS Level 1 compliant card processing, tokenization, and multi-acquirer routing systems.",
     ["Design payment processing services with idempotency keys and sub-100ms authorization latency.", "Implement card tokenization and secure credit card vaulting adhering to PCI-DSS standards.", "Optimize smart transaction routing across global acquiring banks to minimize processing fees.", "Manage payment webhook delivery and automated chargeback reconciliation."],
     ["6+ years building FinTech or payment processing infrastructure.", "Deep understanding of card network rails (Visa, Mastercard), ISO 8583, and tokenization.", "Strong distributed systems design in Java, Go, or Python.", "Proven record of PCI-DSS audit compliance."]),

    ("healthtech_hipaa_engineer", "HealthTech Data Platform Engineer (HIPAA / FHIR)", "Healthcare Informatics", "MediCloud Health",
     "Build secure, HIPAA-compliant patient health record integration and analytics pipelines using HL7 FHIR standards.",
     ["Develop health data integration pipelines conforming to HL7 FHIR (Fast Healthcare Interoperability Resources).", "Implement strict data-at-rest and data-in-transit encryption, audit logging, and PHI de-identification.", "Integrate with Electronic Health Record (EHR) systems (Epic, Cerner) via SMART-on-FHIR APIs.", "Ensure continuous compliance with HIPAA Security Rules and SOC 2 audits."],
     ["4+ years in healthcare software engineering.", "Hands-on experience with HL7, FHIR, and EHR integrations.", "Strong familiarity with HIPAA compliance and Protected Health Information (PHI) handling.", "Proficiency in Python, Java, or C#."]),

    ("ecommerce_high_concurrency_engineer", "E-Commerce High-Concurrency Platform Engineer", "Core Storefront", "PrimeCart Commerce",
     "Build flash-sale checkout engines capable of handling 200,000 concurrent checkout requests during Black Friday.",
     ["Design inventory reservation locking mechanisms utilizing Redis distributed locks (Redlock).", "Optimize checkout database transactions to prevent overselling while maximizing throughput.", "Implement asynchronous order processing queues with Kafka and dead-letter handling.", "Conduct rigorous peak load testing and chaos injection ahead of annual shopping holidays."],
     ["5+ years in high-traffic e-commerce or transactional systems.", "Deep knowledge of database concurrency, optimistic locking, and distributed caching.", "Proficiency in Go, Java, or Node.js.", "Proven track record handling Black Friday scale traffic."]),

    ("autonomous_robotics_ros_engineer", "Robotics Software Engineer (ROS 2 / Navigation)", "Autonomous Robotics", "RoboFleet Dynamics",
     "Develop autonomous mobile robot (AMR) navigation, path planning, and obstacle avoidance using ROS 2 and C++.",
     ["Implement SLAM (Simultaneous Localization and Mapping) and trajectory planning algorithms in ROS 2.", "Optimize costmap calculations and dynamic obstacle avoidance for warehouse environments.", "Interface with hardware sensors (LiDAR, ultrasonic, depth cameras) over CAN bus and Ethernet.", "Simulate warehouse robotic fleets using Gazebo before real-world deployment."],
     ["3+ years experience with ROS / ROS 2 and modern C++.", "Strong understanding of kinematics, path planning (A*, Dijkstra, DWA), and SLAM.", "Experience with Gazebo simulation and sensor integration.", "Hands-on debugging on physical robotic hardware."]),

    ("edtech_streaming_engineer", "EdTech Real-Time Streaming Engineer (WebRTC)", "Live Learning", "EduStream Live",
     "Build ultra-low-latency interactive classroom streaming infrastructure for synchronous global online learning.",
     ["Deploy selective forwarding units (SFU) utilizing Mediasoup or Pion WebRTC in Go.", "Implement adaptive bitrate (ABR) streaming adjusting to low-bandwidth mobile connections.", "Build interactive classroom features: live whiteboard synchronization, chat, and breakout rooms.", "Monitor real-time packet loss, jitter, and frame rate metrics across global regions."],
     ["4+ years in video streaming or real-time communications.", "Deep understanding of WebRTC protocols, ICE, STUN/TURN, and RTP/RTCP.", "Proficiency in Go, C++, or Node.js.", "Experience with scalable media server deployments."]),

    ("adtech_low_latency_engineer", "AdTech Real-Time Bidding (RTB) Engineer", "Ad Platform", "BiddingSpeed Networks",
     "Build real-time bidding auction engines evaluating bid requests within a strict 20-millisecond hard deadline.",
     ["Design ultra-fast auction decision microservices in C++ or Rust responding in under 15ms.", "Implement high-throughput in-memory user profile lookup using Aerospike or Dragonfly.", "Optimize network packet serialization and connection reuse across global ad exchanges.", "Analyze click fraud patterns and anomalous impression spam."],
     ["4+ years in AdTech or real-time trading systems.", "Mastery of C++, Rust, or ultra-optimized Go.", "Experience with OpenRTB 2.5 / 3.0 specifications.", "Obsessive focus on microsecond latency profiling."]),

    ("iot_device_telemetry_engineer", "IoT Cloud & Telemetry Platform Engineer", "Connected Devices", "IoTLab Global",
     "Build cloud platforms ingesting telemetry from 5 million connected smart home sensors over MQTT.",
     ["Deploy scalable MQTT broker clusters (EMQX, VerneMQ) processing 100,000 messages per second.", "Ingest time-series telemetry into TimescaleDB or ClickHouse for real-time sensor analytics.", "Implement device authentication using X.509 certificates and secure device shadow synchronization.", "Build automated firmware update (OTA) campaign management tools."],
     ["4+ years in IoT cloud architecture and device telemetry.", "Deep knowledge of MQTT, CoAP, and HTTP/2 protocols.", "Experience with time-series databases (TimescaleDB, InfluxDB, ClickHouse).", "Python, Go, or Java backend proficiency."]),

    ("supply_chain_logistics_engineer", "Supply Chain Systems & Routing Engineer", "Logistics Tech", "LogiTrack Solutions",
     "Develop vehicle routing optimization and real-time inventory tracking engines for global freight networks.",
     ["Implement vehicle routing problem (VRP) optimization algorithms using Google OR-Tools and Python.", "Track GPS fleet movements and calculate dynamic Estimated Time of Arrival (ETA) updates.", "Integrate with third-party logistics APIs, customs databases, and barcode warehouse scanners.", "Build real-time tracking dashboards for logistics dispatchers."],
     ["4+ years in logistics software or operational research.", "Experience with combinatorial optimization (OR-Tools, genetic algorithms).", "Strong Python and spatial database skills (PostGIS).", "Familiarity with supply chain EDI protocols."]),

    ("legaltech_document_ai_engineer", "LegalTech Document Intelligence Engineer", "Legal Intelligence", "LexiAI Solutions",
     "Build specialized document parsing, contract clause extraction, and semantic search systems for law firms.",
     ["Train and fine-tune OCR and layout analysis models on dense legal contracts and court filings.", "Implement specialized RAG pipelines cross-referencing statutory codes and legal precedents.", "Extract key contract obligations, indemnification clauses, and expiration dates with high precision.", "Ensure strict legal privilege data segregation and client matter confidentiality."],
     ["3+ years in NLP, document understanding, or LegalTech applications.", "Experience with document parsing (LayoutLM, PDFMiner, PyMuPDF) and RAG.", "Understanding of legal document structures and vocabulary.", "High bar for factual accuracy and zero hallucination."]),

    ("telecom_5g_network_engineer", "Telecom 5G Core Network Software Engineer", "Telco Cloud", "NovaTel Networks",
     "Develop cloud-native 5G core network functions (AMF, SMF, UPF) deployed on containerized telco clouds.",
     ["Implement 3GPP compliant 5G service-based architecture (SBA) network functions in C++ or Go.", "Optimize User Plane Function (UPF) packet forwarding utilizing DPDK (Data Plane Development Kit).", "Deploy telco cloud infrastructure on Kubernetes with SR-IOV and DPDK CNI plugins.", "Participate in interoperability testing with major telecom equipment vendors."],
     ["4+ years in telecommunications or network protocol engineering.", "Deep knowledge of 3GPP 5G specifications and protocol stacks (HTTP/2, PFCP, GTP-U).", "C++ or Go programming with DPDK experience.", "Understanding of network virtualization (NFV, SDN)."]),

    ("biotech_bioinformatics_pipeline_dev", "Bioinformatics Pipeline & NGS Engineer", "Genomics Core", "GenomeTech Labs",
     "Build automated Next-Generation Sequencing (NGS) data processing pipelines on cloud HPC clusters.",
     ["Author reproducible genomic variant calling workflows using Nextflow, Snakemake, and Docker.", "Process raw FASTQ / BAM sequencing files using GATK, BWA, and Samtools.", "Orchestrate high-performance computing (HPC) batch workloads on AWS Batch and Slurm.", "Collaborate with computational biologists to analyze rare disease genetic variants."],
     ["3+ years in bioinformatics pipeline development or computational genomics.", "Expertise in Nextflow or Snakemake workflow orchestrators.", "Familiarity with genomics tools (GATK, BWA, VCF tools).", "Strong Python and Bash scripting skills."]),

    ("gaming_multiplayer_cloud_architect", "Gaming Cloud Infrastructure Architect", "Game Operations", "EpicPlay Studios",
     "Architect cloud infrastructure autoscaling dedicated game servers globally to match player concurrency spikes.",
     ["Build multi-region game server autoscaling infrastructure using Agones and Google Kubernetes Engine (GKE).", "Implement automated DDoS protection and UDP traffic proxying using AWS Shield and Cloudflare Magic Transit.", "Deploy global Redis caching clusters for player inventory and cross-platform friend lists.", "Conduct stress testing ahead of major game season launches."],
     ["5+ years cloud infrastructure experience with dedicated focus on gaming or real-time workloads.", "Kubernetes, Agones, Terraform, AWS or GCP.", "Deep understanding of UDP networking, latency jitter, and DDoS mitigation.", "Passion for online gaming."]),

    ("aerospace_avionics_software_engineer", "Avionics Software Verification Engineer (DO-178C)", "Aerospace Systems", "AeroDynamics Corp",
     "Develop and verify safety-critical flight control software adhering to FAA DO-178C Level A standards.",
     ["Author deterministic C code for flight management and autopilot sensor processing.", "Execute structural coverage analysis (MC/DC - Modified Condition/Decision Coverage) on embedded code.", "Develop hardware-in-the-loop (HIL) automated test benches simulating flight dynamics.", "Participate in formal FAA certification audits and safety reviews."],
     ["4+ years in safety-critical software development (avionics, automotive, or medical devices).", "Hands-on experience with DO-178C or ISO 26262 compliance.", "Proficiency in embedded C, static analysis tools (Polyspace), and requirement traceability.", "Uncompromising dedication to safety and verification."]),

    ("insurance_claim_analytics_engineer", "InsurTech Claim Intelligence Engineer", "Insurance Analytics", "CoverSure Technologies",
     "Build predictive claims adjudication, fraud scoring, and computer vision damage assessment pipelines.",
     ["Deploy computer vision models analyzing vehicle damage photos to automate repair cost estimation.", "Build automated fraud detection pipelines analyzing claim text, metadata, and historical records.", "Implement explainable AI models (SHAP values) to provide clear rationales for claim risk scores.", "Integrate with core insurance policy administration platforms."],
     ["3+ years in machine learning or data science within insurance or finance.", "Experience with computer vision (image classification, segmentation) and tabular ML (LightGBM).", "Focus on model interpretability (SHAP, LIME) and regulatory fairness.", "Python, SQL, and cloud deployment."]),

    ("renewable_energy_smart_grid_dev", "Smart Grid & Energy Analytics Engineer", "CleanTech Intelligence", "VoltGrid Systems",
     "Develop predictive time-series models for solar/wind energy generation forecasting and battery dispatch optimization.",
     ["Build time-series forecasting models (Temporal Fusion Transformers, Prophet) for renewable power output.", "Optimize battery energy storage system (BESS) charge/discharge schedules against dynamic electricity pricing.", "Ingest supervisory control and data acquisition (SCADA) telemetry over DNP3 and Modbus protocols.", "Deliver automated grid stability reporting to utility operators."],
     ["3+ years in energy analytics, smart grid tech, or industrial IoT.", "Strong time-series forecasting experience in Python.", "Understanding of electrical grid fundamentals and wholesale electricity markets.", "Experience with industrial communication protocols."]),

    ("automotive_infotainment_dev", "Automotive Infotainment Software Engineer (Android Automotive)", "Connected Car", "DriveOS Mobility",
     "Build responsive in-vehicle infotainment (IVI) and digital cockpit software using Android Automotive OS (AAOS).",
     ["Develop native in-vehicle applications in Kotlin compliant with driver distraction safety guidelines.", "Interface with vehicle hardware buses via Vehicle Hardware Abstraction Layer (VHAL).", "Optimize touch responsiveness and cold-boot times on automotive-grade Qualcomm Snapdragon chips.", "Execute automated UI tests simulating varied vehicle speeds and driving conditions."],
     ["4+ years in Android software development with experience in Android Automotive OS (AAOS) or AOSP.", "Strong Kotlin and modern Android architecture proficiency.", "Understanding of vehicle networks (CAN, Ethernet) and safety constraints.", "Familiarity with automotive testing frameworks."]),

    ("retail_pos_systems_engineer", "Retail Point-of-Sale (POS) Systems Architect", "Retail Technology", "OmniStore Tech",
     "Build resilient, offline-capable store checkout terminals and omnichannel inventory synchronization systems.",
     ["Architect offline-first desktop POS software with local SQLite caching and eventual sync.", "Integrate with peripheral hardware: barcode scanners, receipt printers, and EMV payment chip terminals.", "Ensure zero duplicate charges during intermittent store internet outages.", "Deploy automated remote software updates across 3,000 retail store terminals."],
     ["5+ years building Point of Sale (POS) or enterprise retail transaction systems.", "C# / .NET, Electron, or Java desktop experience.", "Deep understanding of offline-first synchronization patterns and conflict resolution.", "Experience with payment terminal protocols."]),

    ("govtech_secure_cloud_engineer", "GovTech FedRAMP Cloud Security Engineer", "Public Sector Cloud", "CivicCloud Systems",
     "Architect FedRAMP High and DoD IL5 compliant cloud infrastructure for state and federal government agencies.",
     ["Build automated cloud compliance guardrails adhering to NIST SP 800-53 security controls.", "Manage government isolated cloud regions (AWS GovCloud, Azure Government).", "Automate System Security Plan (SSP) evidence collection and vulnerability scanning.", "Collaborate with government agency security officers during Authority to Operate (ATO) assessments."],
     ["4+ years in cloud security with dedicated FedRAMP or government compliance experience.", "Active or eligible for US Security Clearance preferred.", "Deep knowledge of NIST SP 800-53, FIPS 140-2, and FedRAMP requirements.", "Terraform and AWS GovCloud mastery."]),

    ("cybersecurity_threat_intel_dev", "Cyber Threat Intelligence & OSINT Developer", "Threat Intelligence", "DarkTrace Defense",
     "Build automated web scrapers, dark web monitoring daemons, and threat indicator enrichment pipelines.",
     ["Develop automated threat intelligence collectors ingesting data from dark web forums and paste sites.", "Extract and enrich Indicators of Compromise (IOCs) using MISP, STIX, and TAXII standards.", "Correlate IOCs against corporate asset logs to identify potential active reconnaissance.", "Author daily executive threat briefings summarizing emerging ransomware campaigns."],
     ["3+ years in cyber threat intelligence or security data engineering.", "Strong Python scripting and automated web scraping skills.", "Experience with threat sharing standards (STIX 2.1, TAXII, MISP).", "Understanding of adversary tactics and ransomware ecosystems."]),

    ("travel_hospitality_reservation_dev", "Travel & Hospitality Global Distribution Engineer", "Travel Platform", "GlobalStay Networks",
     "Build high-concurrency hotel and flight booking reservation engines connecting to Global Distribution Systems (GDS).",
     ["Integrate booking APIs with global distribution systems (Amadeus, Sabre) via OTA XML and REST.", "Implement low-latency room availability caching using Redis with sub-50ms search times.", "Ensure transactional consistency during concurrent room booking attempts.", "Manage currency conversion, tax calculation, and multi-language localization."],
     ["4+ years in travel technology or high-volume booking platforms.", "Proficiency in Java, Python, or Go.", "Deep experience with distributed caching (Redis) and relational database transactions.", "Familiarity with GDS or OTA hospitality API standards."]),

    # Additional Roles to exceed 100+ count
    ("flutter_mobile_developer", "Senior Mobile Engineer (Flutter & Dart)", "Mobile Products", "CrossApp Studios",
     "Develop cross-platform iOS and Android consumer mobile applications using Flutter 3+ and Dart.",
     ["Build reactive, clean state-managed user interfaces with Flutter Bloc or Riverpod.", "Integrate native platform channels for camera, Bluetooth, and biometric authentication.", "Optimize rendering frame rate (60/120 FPS) and eliminate UI jank.", "Maintain automated build and release distribution pipelines on Codemagic."],
     ["3+ years dedicated mobile software development with 2+ years in production Flutter.", "Deep knowledge of Dart asynchronous programming (Streams, Futures).", "Experience publishing and maintaining apps on both App Store and Google Play.", "Strong UI/UX design sensibility."]),

    ("react_native_crossplatform_dev", "Senior React Native Engineer", "Mobile Engineering", "OmniMobile Labs",
     "Build scalable cross-platform mobile apps using React Native, TypeScript, and native bridge modules.",
     ["Develop fluid mobile interfaces using React Native and React Navigation.", "Write custom Swift / Kotlin bridge modules for proprietary device hardware interactions.", "Profile memory leaks and optimize JavaScript thread frame rates using Flipper.", "Implement code-push over-the-air updates and telemetry tracking with Bugsnag."],
     ["4+ years software engineering with 2+ years in React Native.", "Solid mastery of TypeScript, React hooks, and mobile lifecycle components.", "Experience with native iOS (Xcode) and Android (Android Studio) build tooling.", "Understanding of offline-first persistence with MMKV or SQLite."]),

    ("kafka_streaming_platform_engineer", "Kafka Platform & Event Streaming Specialist", "Streaming Core", "EventFlow Cloud",
     "Operate, scale, and secure enterprise Apache Kafka clusters handling 50 billion events daily.",
     ["Manage multi-broker Apache Kafka and Schema Registry clusters on Kubernetes using Strimzi.", "Tune broker heap sizing, page cache utilization, and replication throttle limits.", "Implement automated partition rebalancing and topic quota governance.", "Troubleshoot consumer rebalances, poison pill messages, and SSL/SASL authentication."],
     ["4+ years dedicated Apache Kafka operations and development.", "Deep understanding of Kafka storage internals (segment files, log compaction, index files).", "Experience with Kafka monitoring (JMX metrics, Prometheus, Cruise Control).", "Proficiency in Python or Java for event streaming."]),

    ("devops_terraform_specialist", "Infrastructure as Code Specialist (Terraform & Pulumi)", "Cloud Automation", "InfraCode Partners",
     "Standardize multi-environment enterprise cloud automation using modular Terraform and Pulumi.",
     ["Author modular, reusable, versioned Terraform modules adhering to HashiCorp best practices.", "Implement automated module validation and linting using tflint, trivy, and infracost.", "Manage multi-state environments and remote state locking with Terraform Cloud and AWS S3/DynamoDB.", "Conduct peer reviews for infrastructure pull requests and enforce zero-drift policies."],
     ["4+ years dedicated infrastructure-as-code automation.", "HashiCorp Certified: Terraform Associate.", "Deep experience with AWS or GCP resource provisioning.", "Strong shell scripting and GitOps pipeline skills."]),

    ("security_compliance_analyst", "Information Security & Compliance Analyst", "Security Assurance", "TrustShield Governance",
     "Manage continuous security compliance frameworks, vendor risk assessments, and audit readiness.",
     ["Review third-party vendor security postures and analyze SOC 2 reports and SIG questionnaires.", "Coordinate annual penetration testing and track remediation deadlines with engineering managers.", "Update internal security policies and conduct quarterly employee security awareness campaigns.", "Maintain the corporate Risk Register and document security exception requests."],
     ["3+ years in information security, IT audit, or compliance.", "Familiarity with SOC 2, ISO 27001, and NIST frameworks.", "Strong analytical, documentation, and stakeholder interview skills.", "Organized and structured communication."]),

    ("data_warehouse_snowflake_engineer", "Senior Snowflake Data Warehouse Engineer", "Data Architecture", "SnowScale Analytics",
     "Design high-performance dimensional data warehouses, data sharing, and Snowpark transformations.",
     ["Architect scalable data marts and dimensional models on Snowflake utilizing Kimball methodologies.", "Develop automated data ingestion pipelines using Snowpipe and external stages on AWS S3.", "Implement programmatic data transformations using Snowpark Python and stored procedures.", "Optimize virtual warehouse credit consumption through query profile analysis and clustering keys."],
     ["4+ years in data warehousing with 2+ years dedicated to Snowflake.", "SnowPro Core Certification preferred.", "Mastery of advanced SQL and Python.", "Deep experience with data modeling and analytics architectures."]),

    ("postgresql_database_tuning_lead", "High-Scale PostgreSQL Tuning & Reliability Lead", "Data Reliability", "PostgresMasters",
     "Provide expert database performance optimization, connection pool scaling, and query tuning.",
     ["Diagnose and eliminate transaction ID wraparound risks and autovacuum starvation.", "Tune multi-gigabyte queries using EXPLAIN ANALYZE and custom partial/expression indexes.", "Architect zero-downtime database sharding and partitioning strategies using Citus.", "Manage disaster recovery and physical streaming replication topologies."],
     ["6+ years dedicated PostgreSQL database administration and query performance optimization.", "Deep understanding of MVCC, WAL internals, and lock contention modes.", "Experience with PgBouncer connection pooling and Patroni high-availability.", "Linux system internals and disk I/O profiling expertise."]),

    ("kubernetes_operator_developer", "Kubernetes Operator & Controller Developer (Go)", "Cloud Native Core", "OperatorCraft",
     "Build custom Kubernetes Custom Resource Definitions (CRDs) and controllers using Kubebuilder and Go.",
     ["Design declarative Custom Resources and reconciliation controllers using Kubebuilder and Controller-Runtime.", "Manage lifecycle operations (backup, failover, upgrades) for stateful applications on Kubernetes.", "Implement admission webhooks for CRD validation and default mutation.", "Write comprehensive integration tests using envtest and Kuttl."],
     ["3+ years in Go systems programming and Kubernetes API extension.", "Deep knowledge of Kubernetes client-go, informers, workqueues, and reconcilers.", "Experience packaging and distributing operators via OperatorHub and Helm.", "Strong understanding of distributed state management."]),

    ("technical_support_escalation_engineer", "Tier-3 Technical Support Escalation Engineer", "Customer Support", "ResolvCloud",
     "Investigate complex customer edge cases, reproduce software defects, and author hotfix patches.",
     ["Analyze application stack traces, database logs, and network pcaps to triage urgent production issues.", "Reproduce customer edge cases in sandboxed staging environments.", "Collaborate with core software engineers to submit bug fix pull requests.", "Author detailed technical Knowledge Base articles and post-mortem incident documentation."],
     ["3+ years in technical support escalation, QA, or software engineering.", "Ability to read and debug code in Python, JavaScript, or Java.", "Strong database querying skills (SQL) and log analysis (Splunk, Datadog).", "Exceptional customer empathy and written communication under pressure."]),

    ("solutions_engineer_fintech", "Senior FinTech Solutions Engineer (Pre-Sales & Integration)", "Commercial Solutions", "PaySphere Tech",
     "Bridge enterprise sales and engineering by demonstrating payment APIs and guiding customer integrations.",
     ["Deliver high-impact technical demonstrations and architectural walkthroughs for enterprise prospects.", "Guide customer engineering teams through API integration, webhooks, and sandbox testing.", "Author technical RFP/RFI responses and document integration edge cases.", "Provide direct customer feedback to product managers to prioritize feature roadmaps."],
     ["4+ years in solutions engineering, sales engineering, or technical consulting in FinTech.", "Hands-on experience with REST APIs, authentication protocols (OAuth, HMAC), and JSON schemas.", "Strong coding skills in Python, Node.js, or Java for integration proofs-of-concept.", "Persuasive presentation and client leadership skills."]),

    ("bioinformatics_cloud_analyst", "Bioinformatics Cloud & Genomic Data Analyst", "Computational Biology", "HelixData Genomics",
     "Analyze large-scale whole-genome sequencing datasets and run automated cloud bioinformatics pipelines.",
     ["Execute variant calling and RNA-seq expression analysis on AWS Batch using Nextflow.", "Perform statistical association testing (GWAS) on large patient cohort genomic datasets.", "Visualize genomic intervals and gene expression distributions using R (Bioconductor) and Python.", "Curate clinical variant interpretations according to ACMG guidelines."],
     ["MS or PhD in Bioinformatics, Computational Biology, or Human Genetics.", "Strong proficiency in R (Bioconductor) and Python (Pandas, SciPy).", "Familiarity with genomic file formats (VCF, BAM, BED, FASTA).", "Experience with cloud bioinformatics pipelines."]),

    ("embedded_iot_security_engineer", "Embedded Device & IoT Security Specialist", "Hardware Security", "SecureSilicon Labs",
     "Perform hardware teardowns, side-channel analysis, and firmware reverse engineering on connected devices.",
     ["Extract and reverse-engineer firmware images from flash memory chips using SPI/JTAG dumpers.", "Audit embedded C/C++ code against memory corruption vulnerabilities (buffer overflows, format strings).", "Evaluate cryptographic hardware accelerators, secure elements, and secure boot implementations.", "Author hardware security assessment reports with actionable mitigation advice."],
     ["3+ years in hardware hacking, embedded security, or firmware reverse engineering.", "Experience with Ghidra, IDA Pro, and hardware debugging tools (Bus Pirate, logic analyzers).", "Knowledge of ARM architecture and hardware attacks (glitching, side-channel).", "Strong ethical hacking mindset."]),

    ("deep_learning_compiler_engineer", "Deep Learning Compiler Engineer (MLIR / TVM)", "AI Compilers", "TensorSpeed Hardware",
     "Develop optimization passes and hardware code generators for neural network operations.",
     ["Author custom MLIR dialects, transformations, and pattern rewriting passes for tensor algebra.", "Optimize kernel code generation for SIMD vector units, GPUs, and custom AI accelerators.", "Benchmark and reduce memory footprint through graph-level operator fusion and memory planning.", "Collaborate with open-source compiler communities (LLVM, MLIR, Apache TVM)."],
     ["3+ years in compiler engineering or deep learning systems optimization.", "Strong C++17 programming and software engineering discipline.", "Knowledge of LLVM, MLIR, or TVM architectures.", "Deep understanding of hardware memory hierarchies and parallel computing."]),

    ("nlp_prompt_engineer", "Senior Prompt Architect & AI Workflow Designer", "Cognitive Solutions", "AgenticWorkflow Inc",
     "Design complex agentic loops, prompt taxonomies, and multi-step reasoning chains for enterprise knowledge workers.",
     ["Architect multi-step reasoning workflows using ReAct, Chain-of-Thought, and self-consistency prompting.", "Implement structured JSON output enforcement and tool-calling schemas with Pydantic and Instructor.", "Benchmark prompt degradation across model version updates (OpenAI, Anthropic, Gemini).", "Author prompt engineering style guides and train enterprise prompt developers."],
     ["2+ years dedicated prompt design or applied AI development.", "Deep understanding of LLM attention mechanisms, context limits, and instruction-following nuances.", "Python scripting ability to automate prompt evaluation benchmarks.", "Creative, precise linguistic and logical problem-solving ability."]),

    ("cloud_security_soc_engineer", "Cloud Security Operations (SOC) Detection Engineer", "Cyber Detection", "CloudWatchdog Sec",
     "Author high-fidelity detection rules, threat hunt queries, and automated incident containment playbooks.",
     ["Write and tune detection rules in Splunk, Sumo Logic, or Snowflake using Sigma and YARA-L.", "Conduct threat hunting exercises across AWS CloudTrail, VPC Flow Logs, and Okta system logs.", "Automate alert triage and false positive reduction using Python and webhook automations.", "Participate in on-call security incident response shifts with standard operating procedures."],
     ["3+ years in SOC analysis, detection engineering, or cloud security.", "Strong familiarity with AWS CloudTrail, GuardDuty, and Kubernetes audit logs.", "Proficiency with SIEM query languages (SPL, KQL, SQL).", "Analytical mindset for separating malicious anomalies from normal developer activity."])
]

# Write all 105 roles to files
all_roles = ROLES_CATALOG.copy()
for cat_id, title, dept, company, overview, resps, reqs in CATEGORIES_EXTENDED:
    all_roles.append({
        "id": cat_id,
        "title": title,
        "dept": dept,
        "company": company,
        "location": "Multiple Locations / Remote",
        "overview": overview,
        "responsibilities": resps,
        "requirements": reqs
    })

print(f"Total Job Descriptions to write: {len(all_roles)}")

for r in all_roles:
    file_path = JOB_DESCRIPTIONS_DIR / f"{r['id']}.md"
    content = f"""# {r['title']}
**Company**: {r['company']}
**Department**: {r['dept']}
**Location**: {r['location']}

## Role Overview
{r['overview']}

## Key Responsibilities
"""
    for resp in r['responsibilities']:
        content += f"- {resp}\n"
    content += "\n## Required Qualifications\n"
    for req in r['requirements']:
        content += f"- {req}\n"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

print(f"Successfully generated {len(all_roles)} realistic job descriptions in: {JOB_DESCRIPTIONS_DIR}")
