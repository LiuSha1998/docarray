"""
在建模并可能分发数据之后，您通常希望将其存储在某个地方。这就是 DocArray 发挥作用的地方！

顾名思义，文档存储提供了一种无缝的方式来存储文档。无论是本地还是远程，您都可以通过相同的用户界面完成这一切：

💿 在磁盘上，作为本地文件系统中的文件
🪣在AWS S3上
☁论吉纳AI云
文档存储界面允许您将文档推送到多个数据源或从多个数据源拉取文档，所有这些都使用相同的用户界面。

例如，让我们看看它如何与磁盘存储一起使用：
"""

from docarray import BaseDoc, DocList


class SimpleDoc(BaseDoc):
    name: str


docs = DocList[SimpleDoc]([SimpleDoc(name=f"doc_{i}") for i in range(10)])
docs.push("file:///tmp/simple_docs")

docs_pulled = DocList[SimpleDoc].pull("file:///tmp/simple_docs")
print(docs_pulled)
print(docs_pulled[0].name)
