from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.vectordb.pgvector import PgVector
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.storage.postgres import PostgresStorage
from dotenv import load_dotenv 
from agno.models.ollama import Ollama
from agno.knowledge.embedder.ollama import OllamaEmbedder


load_dotenv()

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

embedder = OllamaEmbedder(
    model="nomic-embed-text"
)

vector_db = PgVector(
    table_name="recipes",
    db_url=db_url
)

knowledge_base = PDFUrlKnowledgeBase(
    urls=[
        "https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"
    ],
    vector_db=vector_db,
    embedder=embedder
)

knowledge_base.load(recreate=True)

storage = PgAgentStorage(
    table_name="agent_sessions",
    db_url=db_url
)

agent = Agent(
    model=Ollama(id="llama3.2:latest"),
    knowledge=knowledge_base,
    storage=storage,
    markdown=True
)

agent.print_response(
    "How do I make chicken and galangal in coconut milk soup?",
    stream=True
)