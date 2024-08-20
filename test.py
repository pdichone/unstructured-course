import json, os

from dotenv import load_dotenv


from unstructured_client import UnstructuredClient
from unstructured_client.models import operations, shared

input_filepath = "./data/fake-memo.pdf"

load_dotenv()

client = UnstructuredClient(
    api_key_auth=os.getenv("UNSTRUCTURED_API_KEY"),
    server_url=os.getenv("UNSTRUCTURED_API_URL"),
)

with open(input_filepath, "rb") as f:
    files = shared.Files(content=f.read(), file_name=input_filepath)

req = operations.PartitionRequest(
    shared.PartitionParameters(
        files=files,
        strategy=shared.Strategy.AUTO,
        split_pdf_page=True,
        split_pdf_allow_failed=True,
        split_pdf_concurrency_level=15,
    )
)

try:
    res = client.general.partition(request=req)
    element_dicts = [element for element in res.elements]
    json_elements = json.dumps(element_dicts, indent=2)

    # Print the processed data.
    print(json_elements)

except Exception as e:
    print(e)
