# TODO: Move to correct location in the project structure or remove if not needed

import os
import random
from pinecone import Pinecone

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Choose the index to work with
index_name = "nhs-cdf-index"
index = pc.Index(index_name)

# Describe index stats
stats = index.describe_index_stats()
print("Index stats:", stats)

# Generate a query vector with the correct dimension (384)
query_vector = [random.uniform(-1, 1) for _ in range(384)]

# Query the index
query_results = index.query(vector=query_vector, top_k=5, include_metadata=True, include_values=True)

print("\nQuery results:")
for match in query_results['matches']:
    print(f"ID: {match['id']}, Score: {match['score']}")
    if 'metadata' in match:
        print(f"Metadata: {match['metadata']}")
    else:
        print("No metadata available")
    if 'values' in match:
        print(f"Vector (first 5 elements): {match['values'][:5]}...")
    else:
        print("No vector values available")
    print("---")

# Print all keys in the match dictionary
print("\nAvailable keys in match dictionary:")
if query_results['matches']:
    print(list(query_results['matches'][0].keys()))

# Fetch specific records
record_ids = [match['id'] for match in query_results['matches'][:3]]  # Get first 3 IDs
fetched_records = index.fetch(ids=record_ids)

print("\nFetched records:")
for id, record in fetched_records['vectors'].items():
    print(f"ID: {id}")
    if 'values' in record:
        print(f"Vector (first 5 elements): {record['values'][:5]}...")
    else:
        print("No vector values available")
    if 'metadata' in record:
        print(f"Metadata: {record['metadata']}")
    else:
        print("No metadata available")
    print("---")

# Print all keys in the fetched record dictionary
print("\nAvailable keys in fetched record dictionary:")
if fetched_records['vectors']:
    first_id = next(iter(fetched_records['vectors']))
    print(list(fetched_records['vectors'][first_id].keys()))