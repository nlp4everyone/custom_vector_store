from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.conversions import common_types as types
from llama_index.core.schema import BaseNode,TextNode
from llama_index.core.base.embeddings.base import BaseEmbedding
from typing import Optional, Union, List, Tuple
from uuid import uuid4
# Type
Num = Union[int, float]
# Params
_DEFAULT_UPLOAD_BATCH_SIZE = 64

class QdrantVectorStore():
    def __init__(self,
                 collection_name :str,
                 url :str = "http://localhost:6333",
                 port :int = 6333,
                 grpc_port :int = 6334,
                 prefer_grpc :bool = False,
                 api_key :Optional[str] = None):
        """Init Qdrant client service"""
        assert collection_name, "Collection name must be string"
        self._client = QdrantClient(url = url,
                                    port = port,
                                    grpc_port = grpc_port,
                                    api_key = api_key,
                                    prefer_grpc = prefer_grpc)
        self._collection_name = collection_name

    def __create_collection(self,
                            collection_name :str,
                            embedding_dimension :int,
                            distance :Distance = Distance.COSINE):
        """Create collection with default name"""
        assert collection_name, "Collection name must be a string"
        assert embedding_dimension, "Dimension must be an integer"

        # Create collection if it doesnt exist
        if not self._client.collection_exists(collection_name):
            self._client.create_collection(
                collection_name = collection_name,
                vectors_config = VectorParams(size = embedding_dimension, distance = distance),
            )

    def __get_embeddings(self,
                         texts :list[str],
                         embedding_model : BaseEmbedding,
                         batch_size :int,
                         num_workers :int,
                         show_progress :bool = True):
        """Return embedding from documents"""
        # Set batch size and num workers
        embedding_model.num_workers = num_workers
        embedding_model.embed_batch_size = batch_size
        # Other information
        model_infor = embedding_model.dict()
        callback_manager = embedding_model.callback_manager
        # Return embedding
        return embedding_model.get_text_embedding_batch(texts = texts, show_progress = show_progress)

    def __get_llama_index_payloads(self,documents :list[BaseNode],
                                   embedding_model_name :str,
                                   include_embedding_name :bool = True) -> list[dict]:
        """Construct the payload data with configured format of LlamaIndex datastyle"""

        # Clear private data from payload
        for i in range(len(documents)):
            documents[i].embedding = None
            documents[i].metadata = {}
            documents[i].excluded_embed_metadata_keys = []
            documents[i].excluded_llm_metadata_keys = []
            # Remove metadata in relationship
            for key in documents[i].relationships.keys():
                documents[i].relationships[key].metadata = {}

        # Get payloads
        payloads = [{"_node_content": document.dict(),
                     "_node_type": document.class_name(),
                     "doc_id": document.id_,
                     "document_id": document.id_,
                     "ref_doc_id": document.id_ } for document in documents]

        # Include embedding name if specify
        if include_embedding_name:
            for i in range(len(payloads)): payloads[i].update({"embedding_model_name": embedding_model_name})
        return payloads

    def __insert_points(self,
                        list_embeddings :list[list[float]],
                        list_payloads :list[dict],
                        point_ids: Optional[list[str]] = None,
                        collection_name :Optional[str] = None,
                        batch_size :int = _DEFAULT_UPLOAD_BATCH_SIZE,
                        parallel :int = 1) -> None:
        """Insert point to the collection"""
        # Check size
        if not len(list_embeddings) == len(list_payloads) == len(point_ids):
            raise Exception("Number of embeddings must be equal with number of payloads")

        # Get default points ids
        if point_ids == None:
            point_ids = [str(uuid4()) for i in range(len(list_embeddings))]

        # Upload point
        self._client.upload_points(
            collection_name = collection_name if collection_name != None else self._collection_name,
            points = [
                PointStruct(id = point_ids[i],
                            vector = list_embeddings[i],
                            payload = list_payloads[i] ) for i in range(len(point_ids))
            ],
            batch_size = batch_size,
            parallel = parallel
        )

    def insert_documents(self,
                         documents :list[BaseNode],
                         embedding_model: BaseEmbedding,
                         embedded_batch_size :int = 64,
                         embedded_num_workers :Optional[int] = None) -> None:
        # Get embedding model name
        embedding_model_name = embedding_model.model_name if isinstance(embedding_model.model_name,str) else ""
        # Define payloads
        payloads = self.__get_llama_index_payloads(documents = documents, embedding_model_name = embedding_model_name)
        # Get content and its embedding
        contents = [doc.get_content() for doc in documents]
        embeddings = self.__get_embeddings(texts = contents,
                                           embedding_model = embedding_model,
                                           batch_size = embedded_batch_size,
                                           num_workers = embedded_num_workers)

        # Get embedding dimension
        embedding_dimension = len(embeddings[0])
        # Create collection if doesn't exist!
        self.__create_collection(collection_name = self._collection_name, embedding_dimension = embedding_dimension)

        # Insert vector to collection
        self.__insert_points(list_embeddings = embeddings, list_payloads = payloads)

    def reembedding_with_collection(self,
                                    embedding_model : BaseEmbedding,
                                    collection_name: Optional[str] = None,
                                    upload_batch_size :int = _DEFAULT_UPLOAD_BATCH_SIZE,
                                    embedded_batch_size :int = 64,
                                    embedded_num_workers :int = 4,
                                    show_progress :bool = True):
        """Re-embedding the existed collection to another collection"""
        # Get points from current collection
        points,_ = self.get_points()

        # Check length of points
        if len(points) == 0:
            raise Exception("The list of points is empty!")

        # Backup ids
        points_ids = [str(point.id) for point in points]
        # Backup payload
        payloads = [dict(point.payload) for point in points]

        # Get content
        if "_node_content" in payloads[0].keys():
            # Extract content with LlamaIndex style
            contents = [str(payload['_node_content']['text']) for payload in payloads]
        else:
            contents = []

        # Get embeddings
        embeddings = self.__get_embeddings(texts = contents,
                                           embedding_model = embedding_model,
                                           batch_size = embedded_batch_size,
                                           num_workers = embedded_num_workers,
                                           show_progress = show_progress)
        # Get collection name
        if collection_name == None:
            # When not specify
            id = str(uuid4().fields[-1])[:5]
            collection_name = f"{self._collection_name}_{id}"

        # Get embedding dimension
        embedding_dimension = len(embeddings[0])
        # Create collection if doesn't exist!
        self.__create_collection(collection_name = collection_name, embedding_dimension = embedding_dimension)

        # Insert
        self.__insert_points(collection_name = collection_name,
                             list_embeddings = embeddings,
                             list_payloads = payloads,
                             point_ids = points_ids,
                             batch_size = upload_batch_size)
    def search(self, query_vector: List[Num], similarity_top_k :int = 3):
        if not self._client.collection_exists(self._collection_name):
            raise Exception(f"Collection {self._collection_name} isn't existed!")
        # Return search
        return self._client.search(
            collection_name = self._collection_name,
            query_vector = query_vector,
            limit = similarity_top_k,
        )

    def update_point(self, id, vector):
        result = self._client.update_vectors(
            collection_name = self._collection_name,
            points = [
                models.PointVectors(
                    id = id,
                    vector = vector
                )])
        print(result)

    def retrieve_points(self, ids :list[Union[str,int]]):
        return self._client.retrieve(collection_name = self._collection_name,
                                     ids = ids,
                                     with_vectors = True)
    def collection_info(self) -> types.CollectionInfo:
        return self._client.get_collection(self._collection_name)

    def get_points(self,
                   limit :Optional[int] = "all",
                   with_vector :bool = False) -> Tuple[List[types.Record], Optional[types.PointId]]:
        """Get all the point in the Qdrant collection or with limited amount"""
        # Get total point
        collection_info = self.collection_info()
        total_point = collection_info.points_count

        # Limit if specify
        if limit == "all": limit = total_point
        # Return point
        return self._client.scroll(collection_name = self._collection_name,
                                   limit = limit,
                                   with_vectors = with_vector)

    def set_payload(self, point :list[Union[str,int]]):
        self._client.set_payload(collection_name = self._collection_name,
                                 payload = {},
                                 points = point)