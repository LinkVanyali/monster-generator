"""
Embeds all Monsters Know blog posts into a ChromaDB collection for RAG retrieval.

Install dependencies first:
    pip3 install chromadb langchain-community langchain-core sentence-transformers

Output: ./chroma_db/  (ChromaDB persist directory, shared with the monster catalog)

The blog posts are stored as a second collection ("blog_tactics") alongside the
monster catalog collection ("monsters"). At generation time, both are queried:
  - monsters     → stat block reference (CR, abilities, action economy)
  - blog_tactics → tactical behaviour reference (target priority, positioning, morale)

Usage: python embed_blog_posts.py
"""

import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

INPUT_FILE = "monsters_know_posts.json"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "blog_tactics"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build_search_payload(post: dict) -> str:
    """
    Rich text blob used for semantic similarity. Emphasises creature name, archetype,
    and combat keywords so queries like 'aquatic predator fire creature' surface the
    right posts even when the exact words don't appear in the content.
    """
    categories = ", ".join(post.get("category_names", []))
    tags = ", ".join(post.get("tag_names", []))
    return (
        f"Title: {post['title']}\n"
        f"Archetypes: {categories}\n"
        f"Tags: {tags}\n\n"
        f"{post['content_text']}"
    ).strip()


def main():
    print(f"Loading {INPUT_FILE} …")
    with open(INPUT_FILE, encoding="utf-8") as f:
        posts = json.load(f)
    print(f"  {len(posts)} posts loaded")

    print(f"Loading embedding model ({EMBEDDING_MODEL}) …")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Building documents …")
    documents = []
    for idx, post in enumerate(posts):
        documents.append(Document(
            page_content=build_search_payload(post),
            metadata={
                "slug":           post["slug"],
                "title":          post["title"],
                "date":           post["date"][:10],
                "categories":     ", ".join(post.get("category_names", [])),
                "tags":           ", ".join(post.get("tag_names", [])),
                # Store full content for injection into the generation prompt
                "content_text":   post["content_text"],
            },
            id=f"blog_{idx}",
        ))

    print(f"Embedding and storing {len(documents)} documents into ChromaDB …")
    print(f"  Collection : {COLLECTION_NAME}")
    print(f"  Directory  : {CHROMA_DIR}")
    print("  (This may take a few minutes on first run while the model downloads.)\n")

    # Delete the collection if it already exists so re-runs are idempotent
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        print(f"  Deleting existing '{COLLECTION_NAME}' collection …")
        client.delete_collection(COLLECTION_NAME)

    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    count = db._collection.count()
    print(f"\nDone. {count} documents embedded → {CHROMA_DIR}/{COLLECTION_NAME}")
    print("\nTo query at generation time:")
    print("  results = db.similarity_search(query, k=5)")
    print("  tactical_context = '\\n\\n---\\n\\n'.join(r.metadata['content_text'] for r in results)")


if __name__ == "__main__":
    main()
