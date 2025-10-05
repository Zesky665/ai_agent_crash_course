# Day 6: FAQ Agent - Deployment Ready

This project implements a modular FAQ agent that can search and answer questions about GitHub repository documentation, specifically configured for the DataTalksClub/faq repository.

## Project Structure

```
Day_6/
├── ingest.py           # Data loading and indexing from GitHub repos
├── search_tools.py     # Search tool class for FAQ queries
├── search_agent.py     # Pydantic AI agent configuration
├── logs.py            # Conversation logging functionality
├── main.py            # Command-line interface
├── app.py             # Streamlit web interface
├── pyproject.toml     # Project dependencies
└── logs/              # Conversation logs (auto-generated)
```

## Features

- **GitHub Repository Ingestion**: Downloads and processes markdown files from any GitHub repo
- **Text Search**: Uses minsearch for efficient document retrieval
- **AI Agent**: Powered by Mistral AI with tool use capabilities
- **Conversation Logging**: Automatic logging of all interactions
- **Dual Interfaces**: Both CLI and web UI available

## Setup

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure API Key

The Mistral API key is configured in both `main.py` and `app.py`. For production use, consider using environment variables:

```bash
export MISTRAL_API_KEY="your-api-key-here"
```

Then update the code to use:
```python
API_KEY = os.getenv('MISTRAL_API_KEY')
```

## Usage

### Command-Line Interface

Run the interactive CLI:

```bash
uv run python main.py
```

This provides a simple text-based interface where you can:
- Type questions and receive answers
- Type `stop` to exit
- All interactions are logged to the `logs/` directory

### Web Interface (Streamlit)

Launch the web UI:

```bash
uv run streamlit run app.py
```

Then open your browser to `http://localhost:8501`

Features:
- Beautiful chat interface
- Message history
- Responsive design
- Persistent conversation across questions

## How It Works

### 1. Data Ingestion (`ingest.py`)

- Downloads GitHub repository as ZIP file
- Extracts and parses markdown/mdx files with frontmatter
- Supports optional chunking with sliding window
- Builds searchable index using minsearch

### 2. Search Tools (`search_tools.py`)

- `SearchTool` class encapsulates the search index
- Returns top 5 most relevant documents for queries
- Used as a tool by the AI agent

### 3. Agent Configuration (`search_agent.py`)

- Creates Pydantic AI agent with Mistral model
- Configures system prompt to guide agent behavior
- Instructs agent to:
  - Use search tool before answering
  - Cite sources with GitHub links
  - Provide general guidance when search returns no results

### 4. Logging (`logs.py`)

- Captures complete agent interactions
- Stores in JSON format with timestamps
- Includes: messages, tools used, model info, system prompts
- Configurable log directory via `LOGS_DIRECTORY` env var

## Model Configuration

Currently uses **Mistral Small** (`mistral-small-latest`) via Mistral AI.

To change models, update the `model` parameter in `search_agent.py`:

```python
def init_agent(index, repo_owner, repo_name, model='mistral-small-latest'):
    # ...
    agent = Agent(
        # ...
        model=f'mistral:{model}'
    )
```

Available Mistral models:
- `mistral-small-latest` (recommended for cost/performance)
- `mistral-medium-latest`
- `mistral-large-latest`

## Customization

### Use a Different Repository

Modify the constants in `main.py` or `app.py`:

```python
REPO_OWNER = "YourGitHubUser"
REPO_NAME = "your-repo-name"
```

Update the filter function to select specific files:

```python
def filter(doc):
    return 'your-folder' in doc['filename']
```

### Adjust System Prompt

Edit `SYSTEM_PROMPT_TEMPLATE` in `search_agent.py` to change agent behavior.

### Enable Chunking

For longer documents, enable chunking in the index initialization:

```python
index = ingest.index_data(
    repo_owner,
    repo_name,
    filter=filter,
    chunk=True,
    chunking_params={'size': 2000, 'step': 1000}
)
```

## Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Export dependencies:
   ```bash
   uv export --no-dev > requirements.txt
   ```
3. Deploy via Streamlit Cloud (https://share.streamlit.io/)
4. Add secrets in Streamlit Cloud settings:
   ```
   MISTRAL_API_KEY="your-key-here"
   ```

### Other Platforms

The app can also be deployed to:
- Heroku
- Google Cloud Run
- AWS Elastic Beanstalk
- Any platform supporting Python web apps

## Known Issues

- **Streaming disabled**: Due to Mistral API reference ID format incompatibility with Pydantic AI streaming, responses are shown all at once
- **Memory usage**: Large repositories may require significant memory during indexing

## Dependencies

Main dependencies:
- `pydantic-ai` - Agent framework
- `mistralai` - Mistral AI API client
- `minsearch` - Text search engine
- `streamlit` - Web UI framework
- `python-frontmatter` - Markdown frontmatter parsing
- `requests` - HTTP client

See `pyproject.toml` for complete list.

## License

This is course material for the AI Hero course.

## Resources
- [Pydantic AI Documentation](https://ai.pydantic.dev/)
- [Mistral AI Documentation](https://docs.mistral.ai/)
- [Streamlit Documentation](https://docs.streamlit.io/)
