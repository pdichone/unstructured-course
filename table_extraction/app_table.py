import json, os

from dotenv import load_dotenv
import pprint


from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError

from unstructured.staging.base import dict_to_elements


input_filepath = "../data/embedded-images-tables.pdf"

load_dotenv()

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
    skip_infer_table_types=[],
    pdf_infer_table_structure=True,  # hey api, please infer the table structure - we want to extract tables
)

try:
    resp = client.general.partition(req)
    elements = dict_to_elements(resp.elements)
except SDKError as e:
    print(e)

tables = [el for el in elements if el.category == "Table"]  # filter out only the tables


res = tables[0].text  # get the first table text
pprint.pprint(res)  # print the table text

table_html = tables[0].metadata.text_as_html  # get the table text as html
# pprint.pprint(table_html)  # print the table text as html

# Print the table html
from io import StringIO
from lxml import etree

parser = etree.XMLParser(remove_blank_text=True)
file_obj = StringIO(table_html)
tree = etree.parse(file_obj, parser)
print(etree.tostring(tree, pretty_print=True).decode())

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.chains.summarize import load_summarize_chain

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")
chain = load_summarize_chain(llm, chain_type="stuff")

print("\n==== Summarized Table ====\n")
res = chain.invoke([Document(page_content=table_html)])
print(res["output_text"])
