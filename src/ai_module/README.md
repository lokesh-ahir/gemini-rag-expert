# AI Module - Modular RAG Chatbot with Intent & Model Routing

A self-contained, extractable module for building intelligent circuit design assistants.

## ✅ Features

- **Intent Classification**: Classifies queries into 4 categories
  - `general_query` - Basic questions
  - `edit_schematic` - Modify existing circuits
  - `create_design` - Build new circuits
  - `debug_circuit` - Troubleshoot circuits

- **Model Routing**: Cost-optimized model selection
  - `general_query` → **Gemini Flash** (cheaper)
  - `edit_schematic/create_design/debug_circuit` → **Gemini Neon 4** (more capable)

- **Circuit Building**: Creates JSON circuits from 1800+ Qdrant components
- **Markdown Guides**: Generates step-by-step assembly instructions
- **Chat History**: PostgreSQL storage with workspace/design organization
- **Cost Tracking**: Monitor model usage and expenses

## 📦 Structure

```
ai_module/
├── __init__.py           # Module entry point
├── config.py             # Configuration management
├── main.py               # Main AIModule class
├── README.md             # This file
├── migrations/
│   └── 001_initial_schema.sql  # PostgreSQL schema
├── intent/
│   ├── __init__.py
│   ├── classifier.py     # Intent classification
│   └── schemas.py        # Pydantic schemas
├── routing/
│   ├── __init__.py
│   ├── model_router.py   # Model routing logic
│   └── cost_tracker.py   # Usage cost tracking
├── rag/
│   ├── __init__.py
│   └── circuit_retriever.py  # Qdrant component retrieval
├── circuit/
│   ├── __init__.py
│   ├── builder.py        # Circuit JSON generation
│   └── schemas.py        # Circuit Pydantic schemas
├── guide/
│   ├── __init__.py
│   └── generator.py      # Markdown guide generation
└── chat/
    ├── __init__.py
    ├── models.py          # SQLAlchemy models
    └── history.py         # Chat history service
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install sqlalchemy asyncpg qdrant-client pydantic scikit-learn sentence-transformers
```

### 2. Set Up PostgreSQL

Create a database and run the migration:

```bash
# Create database
createdb ai_module

# Run migration
psql ai_module -f src/ai_module/migrations/001_initial_schema.sql
```

### 3. Configure Environment

```bash
export AI_MODULE_DB_URL="postgresql://user:pass@localhost/ai_module"
export QDRANT_URL="http://localhost:6333"
export GOOGLE_API_KEY="your_google_api_key"
```

### 4. Use the Module

```python
from src.ai_module import AIModule

# Initialize
module = AIModule(
    db_url="postgresql://user:pass@localhost/ai_module",
    qdrant_url="http://localhost:6333",
    google_api_key="your_key"
)

# Create workspace and design
workspace = module.create_workspace("My Workspace")
design = module.create_design(workspace["id"], "Amplifier Circuit")

# Process a chat message (full pipeline)
response = module.process_chat_message(
    workspace_id=workspace["id"],
    design_id=design["id"],
    user_query="Create an amplifier with LM386"
)

print(response["intent"])        # "create_design"
print(response["model"])         # "gemini-neon-4-pro"
print(response["circuit"])       # JSON circuit
print(response["guide"])         # Markdown guide

# Get chat history
messages, total, pages = module.get_chat_messages(
    design_id=design["id"],
    page=1,
    limit=25
)
```

## 🔌 API Reference

### Intent Classification

```python
intent = module.classify_intent("Create an amplifier")
# Returns: IntentClassification(intent="create_design", confidence=0.95, reasoning="...")
```

### Model Routing

```python
routing = module.route_model("create_design")
# Returns: RoutingDecision(model=ModelType.GEMINI_NEON_4, model_id="gemini-neon-4-pro", ...)
```

### Circuit Building

```python
circuit, errors = module.build_circuit("Create amplifier with LM386")
# Returns: (Circuit object, validation_errors)

# Convert to JSON
circuit_json = circuit.model_dump_json()
```

### Component Search

```python
components = module.search_components(
    query="1k resistor",
    component_type="resistor",
    limit=10
)
```

### Guide Generation

```python
guide = module.generate_guide(circuit)
# Returns: Markdown string with step-by-step instructions
```

### Chat History

```python
# Add message
msg = module.add_chat_message(
    workspace_id=workspace_id,
    design_id=design_id,
    role="user",
    content="Create an amplifier",
    intent="create_design",
    model_used="gemini-neon-4-pro",
    circuit_json=circuit_json,
    guide_markdown=guide
)

# Get messages (paginated)
messages, total, pages = module.get_chat_messages(design_id, page=1, limit=25)
```

### Cost Tracking

```python
# Record usage
module.record_usage(
    model_type=ModelType.GEMINI_NEON_4,
    intent="create_design",
    token_count=1000,
    design_id=design_id,
    workspace_id=workspace_id
)

# Get report
report = module.cost_tracker.get_report(last_24_hours=True)
```

## 📊 Database Schema

### Tables

| Table | Purpose |
|-------|---------|
| `workspaces` | Workspace information |
| `designs` | Design information (each has ONE chat) |
| `messages` | Chat messages with intent/model metadata |

### Key Fields

**messages table:**
- `design_id` - Links to design
- `workspace_id` - Links to workspace
- `role` - user/assistant/system
- `content` - Message text
- `intent` - Classified intent
- `model_used` - Model that generated response
- `circuit_json` - JSON circuit (for design messages)
- `guide_markdown` - Markdown guide (for design messages)

## 🔄 Extracting the Module

To use this module in another project:

```bash
# Copy the entire ai_module directory
cp -r src/ai_module /path/to/other/project/

# Install dependencies
pip install sqlalchemy asyncpg qdrant-client pydantic

# Use in your project
from ai_module import AIModule, get_ai_module
```

## 🛠️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_MODULE_DB_URL` | `postgresql://localhost/ai_module` | PostgreSQL connection URL |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | - | Qdrant API key (for cloud) |
| `QDRANT_COLLECTION` | `circuit_components` | Qdrant collection name |
| `GOOGLE_API_KEY` | - | Google API key for Gemini |
| `GEMINI_FLASH_MODEL` | `gemini-1.5-flash` | Flash model ID |
| `GEMINI_NEON4_MODEL` | `gemini-neon-4-pro` | Neon 4 model ID |

### Programmatic Configuration

```python
from src.ai_module.config import AIModuleConfig

config = AIModuleConfig(
    database=AIModuleConfig.DatabaseConfig(url="postgresql://..."),
    qdrant=AIModuleConfig.QdrantConfig(url="http://...", api_key="..."),
    models=AIModuleConfig.ModelConfig(
        google_api_key="...",
        gemini_flash_id="gemini-1.5-flash",
        gemini_neon4_id="gemini-neon-4-pro"
    )
)

module = AIModule(config=config)
```

## 🧪 Testing

Run the tests:

```bash
cd src/ai_module
python -m pytest tests/ -v
```

## 📝 Notes

- Each **design** has exactly **one chat** with multiple messages
- Chat history is organized by `workspace_id` and `design_id`
- Messages are paginated (25 per page by default)
- Circuit JSON follows a standardized schema with components and connections
- Markdown guides include: overview, components list, assembly steps, connections, rationale, testing
- Cost tracking monitors usage by model, intent, workspace, and design

## 🎯 Future Enhancements

- [ ] Add more sophisticated intent classification (ML-based)
- [ ] Implement actual LLM calls for circuit generation
- [ ] Add circuit simulation capabilities
- [ ] Add more component types to Qdrant
- [ ] Add user authentication/authorization
- [ ] Add rate limiting
- [ ] Add caching for frequent queries
