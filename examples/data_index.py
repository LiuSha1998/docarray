"""
索引可让您在矢量数据库中索引文档，以实现基于相似性的高效检索。
支持 ANN 向量搜索、文本搜索、过滤和混合搜索。
目前，文档索引支持Weaviate、Qdrant、ElasticSearch、 Redis和HNSWLib
"""

from docarray import BaseDoc, DocList
from docarray.index import HnswDocumentIndex
from docarray.typing import ImageUrl, ImageTensor, NdArray
import numpy as np


class ImageDoc(BaseDoc):
    url: ImageUrl
    tensor: ImageTensor
    embedding: NdArray[128]


dl = DocList[ImageDoc](
    [
        ImageDoc(
            url="https://upload.wikimedia.org/wikipedia/commons/2/2f/Alpamayo.jpg",
            tensor=np.zeros((3, 224, 224)),
            embedding=np.random.random((128,)),
        )
        for _ in range(100)
    ]
)

# 创建索引
index = HnswDocumentIndex[ImageDoc](work_dir="/tmp/test_index")
# index data
index.index(dl)
# 查询
query = dl[0]
results, scores = index.find(query, limit=10, search_field="embedding")
