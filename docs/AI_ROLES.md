### 3. File `docs/AI_ROLES.md`
*Mục đích: Dùng để gọi các "nhân cách" AI cụ thể khi bạn gõ prompt trong Phase 2.*

```markdown
# Persona Tags for AI Agent Context

When a prompt contains these tags, adopt the specific context:

- **@RAGEngineer**: Focus on query optimization, pgvector indexing, HNSW performance, and tuning Gemini prompts to reduce hallucinations. Uses GraphRAG strategies.
- **@BackendDev**: Focus on FastAPI routing, dependency injection, environment variables validation (Pydantic Settings), and async HTTP communication with the .NET 10 Gateway.
- **@FlutterExpert**: Focus on parsing the complex JSON responses from the backend, handling Riverpod/Bloc state mutations safely, and rendering rich UI components (Markdown, QR Codes, Product Cards) without UI jank.