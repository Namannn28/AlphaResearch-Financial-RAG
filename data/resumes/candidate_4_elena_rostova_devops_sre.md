# Elena Rostova
**Role**: Lead Site Reliability Engineer & Platform Architect
**Email**: elena.rostova@example.com | **Location**: Chicago, IL | **GitHub**: github.com/erostova-infra

---

## Professional Summary
Site Reliability Engineer and Cloud Architect with 8+ years of expertise in Kubernetes cluster management, infrastructure-as-code (Terraform, Pulumi), observability, and zero-downtime high-availability systems. Champion of automated chaos engineering, SLA/SLO definition, and GitOps workflows.

---

## Technical Skills
- **Containerization & Orchestration**: Kubernetes, Docker, Helm, ArgoCD, Flux, Istio Service Mesh
- **Infrastructure as Code**: Terraform, Pulumi, Ansible, Packer, AWS CloudFormation
- **Cloud Providers**: AWS (Multi-account Organizations, IAM, VPC, EKS, KMS), GCP (GKE, Cloud Armor)
- **Observability**: Prometheus, Thanos, Grafana, OpenTelemetry, Datadog, ELK Stack, Jaeger
- **Security & Compliance**: HashiCorp Vault, Falco, Trivy, Kyverno, SOC 2 Type II Compliance

---

## Work Experience

### Staff Site Reliability Engineer | StripeGuard Financial (2021 – Present)
- Designed and maintained 14 production Kubernetes clusters hosting over 220 PCI-DSS Level 1 compliant microservices across multi-region AWS infrastructure.
- Established strict SLOs/SLIs (Service Level Objectives) for critical payment path, sustaining 99.995% availability through automated circuit-breakers and proactive capacity provisioning.
- Implemented GitOps deployment pipelines via ArgoCD, standardizing declarative rollouts and slashing deployment rollback duration from 25 minutes to under 40 seconds.
- Introduced Chaos Engineering drills using Chaos Mesh, uncovering 11 critical cascading failure vulnerabilities prior to peak Black Friday traffic.

### Senior DevOps Engineer | OmniCloud Tech (2018 – 2021)
- Migrated legacy bare-metal infrastructure to Terraform-managed AWS EKS clusters, reducing infrastructure provisioning cycle time from 3 weeks to 18 minutes.
- Built centralized logging and tracing infrastructure using OpenTelemetry collectors and Grafana Tempo, cutting MTTR (Mean Time to Resolution) from 55 minutes to 12 minutes.
- Automated secret rotation and cryptographic key lifecycle management using HashiCorp Vault and AWS KMS.

---

## Key Projects

### GitOps-KubeAudit
- Open-source Kubernetes admission controller validating resource requests, security context constraints, and network policies prior to admission into production namespaces.

### Multi-Cloud Disaster Recovery Failover Controller
- Engineered an automated DNS and database failover automation daemon with AWS Route 53 and CockroachDB across 3 geographic availability zones.

---

## Education
- **Bachelor of Science in Systems Engineering** | University of Illinois Urbana-Champaign (2014 – 2018)
- **Certified Kubernetes Administrator (CKA)** | Linux Foundation
- **HashiCorp Certified: Terraform Associate**
