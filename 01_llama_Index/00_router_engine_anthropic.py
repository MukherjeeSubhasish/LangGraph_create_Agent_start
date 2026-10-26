# %% [markdown]
# Agentic RAG with LlamaIndex — Lesson 1: Router Query Engine (Claude version)
#
# Setup (run inside your venv):
#   uv pip install llama-index-core llama-index-readers-file llama-index-llms-anthropic \
#                  llama-index-embeddings-huggingface nest_asyncio python-dotenv
#   curl -L https://arxiv.org/pdf/2308.00352 -o metagpt.pdf     # the MetaGPT paper
#   .env needs: ANTHROPIC_API_KEY=sk-ant-...   (no OpenAI key needed)

# %%
import nest_asyncio
from dotenv import load_dotenv

nest_asyncio.apply()
_ = load_dotenv()

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    SummaryIndex,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# %% LLM + embeddings
# Claude for the LLM; Anthropic has no embedding model, so use a free local one.
# (First run downloads ~130 MB for the embedding model.)
Settings.llm = Anthropic(model="claude-sonnet-5-5", max_tokens=2048)
# Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")


# %% Build the router engine (replaces the course's utils.get_router_query_engine)
def get_router_query_engine(file_path, verbose=True):
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    # nodes = SentenceSplitter(chunk_size=1024).get_nodes_from_documents(documents)
    nodes = SentenceSplitter(chunk_size=512).get_nodes_from_documents(documents)

    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes)

    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    # vector_query_engine = vector_index.as_query_engine(similarity_top_k=5)
    vector_query_engine = vector_index.as_query_engine(similarity_top_k=8)

    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description="Useful for summarization questions related to MetaGPT",
    )
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description="Useful for retrieving specific context from the MetaGPT paper.",
    )

    return RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[summary_tool, vector_tool],
        verbose=verbose,
    )


query_engine = get_router_query_engine("metagpt.pdf")

# %% Summary question -> should route to summary_tool
response = query_engine.query("What is the summary of the document?")
print(str(response))
print(len(response.source_nodes))

# %% Specific question -> should route to vector_tool
response = query_engine.query("How do agents share information with other agents?")
print(str(response))

# %%
response = query_engine.query("Tell me about the ablation study results?")
print(str(response))