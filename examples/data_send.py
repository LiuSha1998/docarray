"""
数据传输
"""

from docarray import BaseDoc
from docarray.typing import ImageTorchTensor
import torch


# model your data
class MyDocument(BaseDoc):
    description: str
    image: ImageTorchTensor[3, 2, 2]


# create a Document
doc = MyDocument(
    description="This is a description",
    image=torch.zeros((3, 2, 2)),
)

# serialize it!
proto = doc.to_protobuf()
# print("proto=", proto)
# input("Press Enter to continue...")
bytes_ = doc.to_bytes()
# print("bytes_=", bytes_)
# input("Press Enter to continue...")
json = doc.json()
print("json=", json)
input("Press Enter to continue...")

# deserialize it!
doc_2 = MyDocument.from_protobuf(proto)
print("doc_2=", doc_2)
input("Press Enter to continue...")
doc_4 = MyDocument.from_bytes(bytes_)
print("doc_4=", doc_4)
input("Press Enter to continue...")
doc_5 = MyDocument.parse_raw(json)
print("doc_5=", doc_5)
