import json, os

from dotenv import load_dotenv
from unstructured_client import UnstructuredClient
from unstructured.partition.auto import partition
from unstructured.partition.pptx import partition_pptx


load_dotenv()

input_filepath = "../data/msft_openai.pptx"

pptx_elements = partition_pptx(filename=input_filepath)
element_dict = [el.to_dict() for el in pptx_elements]

print(json.dumps(element_dict, indent=2))

# client = UnstructuredClient(
#     api_key_auth=os.getenv("UNSTRUCTURED_API_KEY"),
#     server_url=os.getenv("UNSTRUCTURED_API_URL"),
# )

# elements = partition(filename=input_filepath)
# element_dict = [el.to_dict() for el in elements]

# print(json.dumps(element_dict, indent=2))
