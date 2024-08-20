import json
import os
import pprint
from dotenv import load_dotenv
import chromadb

from sqlalchemy import JSON
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError
from unstructured.staging.base import dict_to_elements
from unstructured.chunking.title import chunk_by_title


input_filepath = "../data/mindset.pdf"

load_dotenv()

client = UnstructuredClient(
    api_key_auth=os.getenv("UNSTRUCTURED_API_KEY"),
    server_url=os.getenv("UNSTRUCTURED_API_URL"),
)

# Initialize Chroma DB
chroma_client = chromadb.PersistentClient(
    path="chroma_tmp", settings=chromadb.Settings(allow_reset=False)
)

# Check if the collection already exists and is populated
collection_name = "mindset"
collection = chroma_client.get_or_create_collection(
    name=collection_name, metadata={"hnsw:space": "cosine"}
)

if collection.count() > 0:
    # Collection already has documents, skip extraction and proceed with retrieval
    print("Collection already exists with documents. Skipping extraction.")
else:
    # Collection is empty, proceed with extraction
    print("Collection is empty. Proceeding with extraction.")

    with open(input_filepath, "rb") as f:
        files = shared.Files(content=f.read(), file_name=input_filepath)

    # remember: the partitioning and extraction may take a while!
    req = shared.PartitionParameters(files=files)
    try:
        resp = client.general.partition(req)
        pprint.pprint(json.dumps(resp.elements[0:3], indent=2))

        chapters = [
            "Embracing a Growth Mindset",
            "Strategies for Cultivating a Growth Mindset",
            "I N T R O D U C T I O N",
            "M I N D S E T",
            "T H E D R I V E R",
            "Growth vs. Fixed Mindset F I X E D",
            "F I X E D",
            "Activities",
        ]
        chapter_ids = {}
        for element in resp.elements:
            for chapter in chapters:
                if element["text"] == chapter and element["type"] == "Title":
                    chapter_ids[element["element_id"]] = chapter
                    break
        print("==== chapters IDs: \n")
        pprint.pprint(chapter_ids)
        print("==== Elements with parent ID>>>: \n")
        chapter_to_id = {v: k for k, v in chapter_ids.items()}
        res = [
            x
            for x in resp.elements
            if x["metadata"].get("parent_id")
            == chapter_to_id["Embracing a Growth Mindset"]
        ]
        pprint.pprint(json.dumps(res, indent=2))

        # Add elements to the Chroma collection
        for element in resp.elements:
            parent_id = element["metadata"].get("parent_id")
            chapter = chapter_ids.get(parent_id, "")
            collection.add(
                documents=[element["text"]],
                ids=[element["element_id"]],
                metadatas=[{"chapter": chapter}],
            )
        print("Documents have been added to the Chroma collection.")

        # Chunking content
        print("Chunking content...")
        elements = dict_to_elements(resp.elements)
        chunks = chunk_by_title(
            elements, combine_text_under_n_chars=100, max_characters=300
        )
        print("==== Chunks: \n")
        # print(chunks)
        re = json.dumps(chunks[0].to_dict(), indent=2)
        print(re)

        print("elements:", len(elements))
        print("chunks:", len(chunks))

    except SDKError as e:
        print(e)

# Perform a hybrid search with metadata
result = collection.query(
    query_texts=["A growth mindset believes in what?"],
    n_results=2,
    where={"chapter": "Embracing a Growth Mindset"},
)
print("\n==== Query Results ==== \n")
pprint.pprint(json.dumps(result, indent=2))
