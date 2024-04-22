"""
docarray表示数据示例
"""

from docarray import BaseDoc
from docarray.typing import TorchTensor, ImageUrl, AudioUrl
from typing import Optional
import torch


# # Define your data model
# class MyDocument(BaseDoc):
#     description: str
#     image_url: ImageUrl  # could also be VideoUrl, AudioUrl, etc.
#     image_tensor: Optional[
#         TorchTensor[1704, 2272, 3]
#     ] = None  # could also be NdArray or TensorflowTensor
#     embedding: Optional[TorchTensor] = None


# 1. 自定义
class WordDocument(BaseDoc):
    description: str
    image_url: ImageUrl  # could also be VideoUri, AudioUri, etc.
    audio_url: Optional[AudioUrl] = None
    image_tensor: Optional[TorchTensor[1704, 2272, 3]] = (
        None  # could also be NdArray or TensorflowTensor
    )
    image_embedding: Optional[TorchTensor] = None
    audio_tensor: Optional[TorchTensor] = (
        None  # 如果用audio的特征来匹配读音相近单词呢，可是音标就可以直接匹配啊。。。这个特征是否有用呢？
    )
    audio_embedding: Optional[TorchTensor] = None
    sense: str
    id: int


# # Create a document
# doc = MyDocument(
#     description="This is a photo of a mountain",
#     image_url="https://upload.wikimedia.org/wikipedia/commons/2/2f/Alpamayo.jpg",
# )


word_doc = WordDocument(
    description="This is a photo of a mountain",
    image_url="https://upload.wikimedia.org/wikipedia/commons/2/2f/Alpamayo.jpg",
    audio_url=None,
    sense="mountain",
    id=1,
)

# Load image tensor from URL
# doc.image_tensor = doc.image_url.load()
word_doc.image_tensor = word_doc.image_url.load()


# Compute embedding with any model of your choice
def clip_image_encoder(image_tensor: TorchTensor) -> TorchTensor:  # dummy function
    return torch.rand(512)


# doc.embedding = clip_image_encoder(doc.image_tensor)

# print(doc.embedding.shape)  # torch.Size([512])

word_doc.image_embedding = clip_image_encoder(word_doc.image_tensor)
print(word_doc.image_embedding)

# 2.嵌套文档
from docarray import BaseDoc
from docarray.documents import ImageDoc, TextDoc
import numpy as np


class MultiModalDocument(BaseDoc):
    image_doc: ImageDoc
    text_doc: TextDoc


doc = MultiModalDocument(
    image_doc=ImageDoc(tensor=np.zeros((3, 224, 224))), text_doc=TextDoc(text='hi!')
)

# 3.多个文档组合
# DocVec: 向量 Documents , documents 中的所有张量都堆叠成一个张量,适合批处理和ML模型内使用
# DocList: 列表 Documents ，documents 中的所有张量均保持原样。非常适合数据流式传输、重新排名和洗牌
# 3.1 DocVec
from docarray import DocVec, BaseDoc
from docarray.typing import AnyTensor, ImageUrl
import numpy as np


class Image(BaseDoc):
    url: ImageUrl
    tensor: AnyTensor  # this allows torch, numpy, and tensor flow tensors


doc_vec = DocVec[Image](
    [
        Image(
            url="https://upload.wikimedia.org/wikipedia/commons/2/2f/Alpamayo.jpg",
            tensor=np.zeros((3, 224, 224)),
        )
        for _ in range(100)
    ]
)  # the DocVec is parametrized by your personal schema!
tensor = doc_vec.tensor  # 提取DocVec中的所有张量
print(tensor.shape)  # torch.Size([100, 3, 224, 224])
print(doc_vec.url)  # 其他的属性也堆叠


# 3.2 DocList, 有append, insert, delete等操作
from docarray import DocList, BaseDoc

doc_list = DocList[Image](
    [
        Image(
            url="https://upload.wikimedia.org/wikipedia/commons/2/2f/Alpamayo.jpg",
            tensor=np.zeros((3, 224, 224)),
        )
        for _ in range(100)
    ]
)
# 仍然可以批量访问
doc_list_tensor = doc_list.tensor  # 提取DocList中的所有张量
print(type(doc_list_tensor))  # list
print(doc_list.url)

# append
doc_list.append(
    Image(
        url="https://upload.wikimedia.org/wikipedia/commons/2/2f/Alpamayo.jpg",
        tensor=np.zeros((3, 224, 224)),
    )
)
# delete
del doc_list[0]
# insert
doc_list.insert(
    0,
    Image(
        url="https://upload.wikimedia.org/wikipedia/commons/2/2f/Alpamayo.jpg",
        tensor=np.zeros((3, 224, 224)),
    ),
)
