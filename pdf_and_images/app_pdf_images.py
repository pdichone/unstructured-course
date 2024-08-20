from dotenv import load_dotenv
import pprint
import os


from unstructured_client import UnstructuredClient
from unstructured_client.models import operations, shared
from unstructured_client.models.errors import SDKError

from unstructured.partition.html import partition_html
from unstructured.partition.pdf import partition_pdf

from unstructured.staging.base import dict_to_elements

load_dotenv()

client = UnstructuredClient(
    api_key_auth=os.getenv("UNSTRUCTURED_API_KEY"),
    server_url=os.getenv("UNSTRUCTURED_API_URL"),
)
input_filepath = "../data/el_nino.html"


html_elements = partition_html(filename=input_filepath)
## Process the Document as HTML
for element in html_elements[:15]:
    pprint.pprint(f"{element.category.upper()}: {element.text}")


print("\n==== PDF ====\n")
input_filepath = "../data/el_nino.pdf"
pdf_elements = partition_pdf(filename=input_filepath, strategy="fast")

for element in html_elements[:15]:
    print(f"{element.category.upper()}: {element.text}")

input_filepath = "../data/el_nino.pdf"
with open(input_filepath, "rb") as f:
    files = shared.Files(
        content=f.read(),
        file_name=input_filepath,
    )

req = shared.PartitionParameters(
    files=files,
    strategy="hi_res",
    hi_res_model_name="yolox",
)

try:
    resp = client.general.partition(req)
    dld_elements = dict_to_elements(resp.elements)
except SDKError as e:
    print(e)


print("\n==== Hi-Res Yolox====\n")

for element in dld_elements[:15]:
    print(f"{element.category.upper()}: {element.text}")


# Compare html, pdf and dld partitioning elements ===
print("\n==== HTML ====\n")
import collections

print(len(html_elements))
html_categories = [el.category for el in html_elements]
res = collections.Counter(html_categories).most_common()
print(res)


print("\n==== DLD -- Document Layout Detection ====\n")
print(len(dld_elements))
dld_categories = [el.category for el in dld_elements]
resu = collections.Counter(dld_categories).most_common()
pprint.pprint(resu)
