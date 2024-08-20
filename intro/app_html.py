from unstructured.partition.html import partition_html
import json

input_filepath = "../data/medium_blog.html"
html_elements = partition_html(input_filepath)

element_dict = [el.to_dict() for el in html_elements]

print(json.dumps(element_dict, indent=2))
