# DineX-AI RAG Chatbot System Architecture

## System Flow Diagram
```mermaid
sequenceDiagram
    participant User as Flutter App (User)
    participant Gateway as .NET 10 Main BE (API Gateway)
    participant AI as FastAPI (DineX-AI Service)
    participant DB as PostgreSQL (pgvector + KG)
    participant Gemini as Google Gemini API

    User->>Gateway: POST /chat (Token, Message)
    Gateway->>Gateway: Auth & Validate
    Gateway->>AI: POST /api/ai/chat (Message, BranchId)
    
    AI->>Gemini: Get Embedding for Message
    Gemini-->>AI: Vector [768]
    
    AI->>DB: Semantic Search (pgvector <=>) + KG JOIN
    DB-->>AI: Top Products & Recommendations
    
    AI->>Gemini: Prompt + Context (Products) + Tools
    Gemini-->>AI: Function Call (e.g., generate_payment_qr)
    
    AI->>Gateway: API Call (Generate QR / Calc Price)
    Gateway-->>AI: QR Data / Price Data
    
    AI->>Gemini: Submit Function Result
    Gemini-->>AI: Final Structured JSON
    
    AI-->>Gateway: ChatResponse JSON
    Gateway-->>User: Rich UI Data (Text, Products, QR)