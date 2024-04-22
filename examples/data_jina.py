"""
Jina 采用 docarray 作为表示和序列化文档的库。

Jina 允许提供使用 DocArray 构建的模型和服务，允许您充分利用 DocArray 的序列化功能来提供和扩展这些应用程序
"""

import numpy as np
from jina import Deployment, Executor, requests
from docarray import BaseDoc, DocList
from docarray.documents import ImageDoc
from docarray.typing import NdArray, ImageTensor


class InputDoc(BaseDoc):
    img: ImageDoc
    text: str


class OutputDoc(BaseDoc):
    embedding_clip: NdArray
    embedding_bert: NdArray


def model_img(img: ImageTensor) -> NdArray:
    return np.random.random((100, 1))


def model_text(text: str) -> NdArray:
    return np.random.random((100, 1))


class MyEmbeddingExecutor(Executor):
    @requests(on="embed")
    def encode(self, docs: DocList[InputDoc], **kwargs) -> DocList[OutputDoc]:
        ret = DocList[OutputDoc]()
        for doc in docs:
            output = OutputDoc(
                embedding_clip=model_img(doc.img.tensor),
                embedding_bert=model_text(doc.text),
            )
            ret.append(output)
        return ret


with Deployment(
    protocols=["http", "grpc"], ports=[12345, 12346], uses=MyEmbeddingExecutor
) as dep:
    resp = dep.post(
        on="/embed",
        inputs=DocList[InputDoc](
            [
                InputDoc(
                    img=ImageDoc(tensor=np.random.random((3, 224, 224))),
                    text="hello",
                )
            ]
        ),
        return_type=DocList[OutputDoc],
    )
    print(resp)
