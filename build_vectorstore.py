import os
from dotenv import load_dotenv
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

load_dotenv(override=True)

DB = "products_vectorstore"
DATASET = "ed-donner/items_full"
CATEGORIES = [
    "Appliances", "Automotive", "Cell_Phones_and_Accessories",
    "Electronics", "Musical_Instruments", "Office_Products",
    "Tools_and_Home_Improvement", "Toys_and_Games"
]

print("Loading dataset from HuggingFace...")
ds = load_dataset(DATASET)
train = ds["train"]

print("Setting up ChromaDB...")
client = chromadb.PersistentClient(path=DB)
if "products" in [c.name for c in client.list_collections()]:
    client.delete_collection("products")
collection = client.get_or_create_collection("products")

print("Loading sentence transformer model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Adding products to vector store (this will take a while)...")
BATCH_SIZE = 500
MAX_ITEMS = 10000

items = list(train.select(range(min(MAX_ITEMS, len(train)))))

for i in tqdm(range(0, len(items), BATCH_SIZE)):
    batch = items[i:i+BATCH_SIZE]
    summaries = [item["summary"] for item in batch if item["summary"]]
    prices = [item["price"] for item in batch if item["summary"]]
    categories = [item["category"] for item in batch if item["summary"]]
    
    if not summaries:
        continue
    
    vectors = model.encode(summaries).tolist()
    ids = [str(i + j) for j in range(len(summaries))]
    metadatas = [{"price": p, "category": c} for p, c in zip(prices, categories)]
    
    collection.add(
        ids=ids,
        embeddings=vectors,
        documents=summaries,
        metadatas=metadatas
    )

print(f"Vector store built with {collection.count():,} products!")