import json, os

import pprint
from dotenv import find_dotenv, load_dotenv


from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError

from unstructured.chunking.title import chunk_by_title
from unstructured.partition.md import partition_md
from unstructured.partition.pptx import partition_pptx
from unstructured.staging.base import dict_to_elements

from langchain_chroma import Chroma


load_dotenv()

input_filepath = "../data/post_ocr.pdf"
persist_directory = "./data/db/chroma/"

client = UnstructuredClient(
    api_key_auth=os.getenv("UNSTRUCTURED_API_KEY"),
    server_url=os.getenv("UNSTRUCTURED_API_URL"),
)

with open(input_filepath, "rb") as f:
    files = shared.Files(content=f.read(), file_name=input_filepath)


req = shared.PartitionParameters(
    files=files,
    strategy="hi_res",
    hi_res_model_name="yolox",
    pdf_infer_table_structure=True,
    skip_infer_table_types=[],
)

try:
    resp = client.general.partition(req)
    pdf_elements = dict_to_elements(resp.elements)
except SDKError as e:
    print(e)

res = pdf_elements[0].to_dict()
# print(res)

# check if there are any tables in the pdf
tables = [el for el in pdf_elements if el.category == "Table"]

# if there are tables, we need to process them
table_html = tables[0].metadata.text_as_html  # get the html of the first table

# Print the table html
from io import StringIO
from lxml import etree

parser = etree.XMLParser(remove_blank_text=True)
file_obj = StringIO(table_html)
tree = etree.parse(file_obj, parser)
# print(etree.tostring(tree, pretty_print=True).decode())

# ===== Filter out other metadata - unwanted content ==
# Get references from the document (show the document to see the references)
reference_title = [
    el for el in pdf_elements if el.text == "References" and el.category == "Title"
]

# Check if we have any matching elements
if len(reference_title) > 0:
    # If there's at least one match, take the first one (assuming there's only one "References" title)
    reference_title = reference_title[0]

    # Convert to dictionary and print
    res = reference_title.to_dict()
    print("\n== Reference Title ==\n")
    pprint.pprint(res)


else:
    print("No element found with the text 'References' and category 'Title'.")


references_id = reference_title.id

# # show the elemente with the reference id
for element in pdf_elements:
    if element.metadata.parent_id == references_id:
        print(element)
        break

# # Filter out the references
pdf_elements = [el for el in pdf_elements if el.metadata.parent_id != references_id]

# # see header first
headers = [el for el in pdf_elements if el.category == "Header"]
res = headers[0].to_dict()
print("\n== Header ==\n")
pprint.pprint(res)

# # # Filter out headers
pdf_elements = [el for el in pdf_elements if el.category != "Header"]

# # =================================
# # ===== Next pptx processing ====
# # =================================
input_filepath = "../data/kg-paulo.pptx"
pptx_elements = partition_pptx(filename=input_filepath)


# # === Process readme file ==
input_filepath = "../data/devops-roadmap.md"
md_elements = partition_md(filename=input_filepath)

# # === Load the document into the database ===
# # 1. chunk the document by title
elements = chunk_by_title(pdf_elements + pptx_elements + md_elements)

# # pip install -U langchain-chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores.utils import filter_complex_metadata

embeddings = OpenAIEmbeddings()

documents = []
for element in elements:
    metadata = element.metadata.to_dict()
    del metadata["languages"]
    metadata["source"] = metadata["filename"]
    documents.append(Document(page_content=element.text, metadata=metadata))

print("\n== Documents ==\n")
pprint.pprint(documents[3])

# vectorstore = Chroma.from_documents(documents, embeddings) # this will give: Try filtering complex metadata from the document using langchain_community.vectorstores.utils.filter_complex_metadata.

vectorstore = Chroma.from_documents(
    filter_complex_metadata(documents), embeddings, persist_directory=persist_directory
)


## load the persisted db
vector_store = Chroma(
    persist_directory=persist_directory, embedding_function=embeddings
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_core.prompts import PromptTemplate


template = """You are an AI assistant specialized in answering questions related to Advancing Post-OCR Correction:
A Comparative Study of Synthetic Data.
You are provided with the following extracted sections from a lengthy document and a related question. Please respond in a conversational manner.
If you are unsure of the answer, simply reply, "Hmm, I'm not sure." Avoid fabricating any responses.
You also know about DevOps roadmaps and Knowledge Graphs from the documents and context you were given.
Question: {input}
=========
{context}
=========
Respond in Markdown:"""

prompt = PromptTemplate.from_template(template)

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

from langchain.chains.combine_documents import create_stuff_documents_chain

document_chain = create_stuff_documents_chain(llm, prompt)

from langchain.chains import create_retrieval_chain

retrieval_chain = create_retrieval_chain(retriever, document_chain)


ocr_res = retrieval_chain.invoke(
    {
        # "input": "what are the three common methods for generating synthetic data in the post-OCR?",
        "input": "tell me about post-OCR domain?",
        "chat_history": [],
    }
)

print("\n== OCR Results ==\n")
pprint.pprint(ocr_res)


# # == Filter retriever dev ops ==
filter_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1, "filter": {"source": "devops-roadmap.md"}},
)
retrieval_chain = create_retrieval_chain(filter_retriever, document_chain)
dev_ops = retrieval_chain.invoke(
    {
        "input": "what are some popular programming languages for DevOps-es? and what resources are available for learning them?",
        # "input": "tell me about post-OCR domain?",
        "chat_history": [],
    }
)
print("\n== DevOps Results ==\n")
pprint.pprint(dev_ops)


# # == Filter retriever Knowledge Graph pptx ==

filter_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1, "filter": {"source": "kg-paulo.pptx"}},
)

retrieval_chain = create_retrieval_chain(filter_retriever, document_chain)

kg_res = retrieval_chain.invoke(
    {
        "input": "give me the main key points about Knowledge Graph?",
        # "input": "tell me about post-OCR domain?",
        "chat_history": [],
    }
)
print("\n== KG Results ==\n")
pprint.pprint(kg_res)
