from functools import lru_cache

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings

from src.config import get_settings


EMBEDDING_DIMENSION = 1024
EMBEDDING_MODEL = "llama-text-embed-v2"


@lru_cache
def get_embeddings():
    s = get_settings()

    if not s.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not configured")

    return PineconeEmbeddings(
        model=EMBEDDING_MODEL,
        dimension=EMBEDDING_DIMENSION,
        pinecone_api_key=s.pinecone_api_key,
    )


def get_pinecone_client():
    s = get_settings()

    if not s.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not configured")

    return Pinecone(api_key=s.pinecone_api_key)


def ensure_index():
    """Create the Pinecone index if needed and verify embedding dimension."""

    s = get_settings()
    pc = get_pinecone_client()

    existing = {x.name for x in pc.list_indexes()}

    if s.pinecone_index_name not in existing:
        pc.create_index(
            name=s.pinecone_index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=s.pinecone_cloud,
                region=s.pinecone_region,
            ),
        )

    else:
        desc = pc.describe_index(s.pinecone_index_name)

        existing_dimension = getattr(desc, "dimension", None)

        if existing_dimension is None and isinstance(desc, dict):
            existing_dimension = desc.get("dimension")

        if (
            existing_dimension
            and int(existing_dimension) != EMBEDDING_DIMENSION
        ):
            raise RuntimeError(
                f"Pinecone index '{s.pinecone_index_name}' has "
                f"dimension {existing_dimension}, but "
                f"{EMBEDDING_MODEL} is configured for "
                f"{EMBEDDING_DIMENSION}-dimensional embeddings."
            )

    return pc.Index(s.pinecone_index_name)


def get_vector_store():
    s = get_settings()

    index = ensure_index()

    return PineconeVectorStore(
        index=index,
        embedding=get_embeddings(),
        namespace=s.pinecone_namespace,
    )


def get_retriever():
    s = get_settings()

    return get_vector_store().as_retriever(
        search_kwargs={"k": s.top_k}
    )