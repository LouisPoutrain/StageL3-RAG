from haystack.document_stores.in_memory import InMemoryDocumentStore
from datasets import load_dataset
from haystack import Document
from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack import Pipeline
from huggingface_hub import login
from haystack.utils import Secret
from haystack.utils.hf import HFGenerationAPIType
import os


document_store = InMemoryDocumentStore()
dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()
docs_with_embeddings = doc_embedder.run(docs)
document_store.write_documents(docs_with_embeddings["documents"])

text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
retriever = InMemoryEmbeddingRetriever(document_store)

template = [
    ChatMessage.from_system("Vous êtes un assistant expert des 7 merveilles du monde."),
    ChatMessage.from_user(
        """Context:
{% for document in documents %}
{{ document.content }}
{% endfor %}
Question : {{question}}"""
    )
]

prompt_builder = ChatPromptBuilder(
    template=template,
    required_variables=["documents", "question"] 
)

api_type = HFGenerationAPIType.SERVERLESS_INFERENCE_API

chat_generator = HuggingFaceAPIChatGenerator(api_type=api_type,
                                        api_params={"model": "HuggingFaceH4/zephyr-7b-beta"},
                                        token=Secret.from_token(os.getenv("HF_HUGGINGFACE_TOKEN"))) # votre token dans .env

basic_rag_pipeline = Pipeline()
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", chat_generator)

basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever.documents", "prompt_builder.documents")  
basic_rag_pipeline.connect("prompt_builder.prompt", "llm.messages")

question = "What does Rhodes Statue look like?"
response = basic_rag_pipeline.run({
    "text_embedder": {"text": question},
    "prompt_builder": {"question": question}
})

print(response["llm"]["replies"][0].text)