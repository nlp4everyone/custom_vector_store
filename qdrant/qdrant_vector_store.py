from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct, ScoredPoint, QueryResponse
from qdrant_client.conversions import common_types as types
from llama_index.core.schema import BaseNode, NodeWithScore, TextNode
from llama_index.core.base.embeddings.base import BaseEmbedding
from typing import Optional, Union, List, Tuple, Sequence, Literal
from uuid import uuid4

# DataType
Num = Union[int, float]
Embedding = List[float]

# Params
_DEFAULT_UPLOAD_BATCH_SIZE = 64

class QdrantVectorStore():
    def __init__(self,
                 collection_name :str,
                 url :str = "http://localhost:6333",
                 port :int = 6333,
                 grpc_port :int = 6334,
                 prefer_grpc :bool = False,
                 api_key :Optional[str] = None,
                 dense_embedding_model: Union[BaseEmbedding, str] = "sentence-transformers/all-MiniLM-L6-v2",
                 spare_embedding_model: Optional[str] = None,
                 embedding_folder_cached: str = "cached",
                 distance: Distance = Distance.COSINE,
                 on_disk: bool = True) -> None:

        """Init Qdrant client service"""
        """- on_disk (bool): Default is True, make sure that original vectors will be stored on disk"""
        assert collection_name, "Collection name must be string"
        self._client = QdrantClient(url = url,
                                    port = port,
                                    grpc_port = grpc_port,
                                    api_key = api_key,
                                    prefer_grpc = prefer_grpc)
        # Get value
        self._collection_name = collection_name
        # Set value
        self._on_disk = on_disk
        self._distance = distance
        self._dense_embedding_model = dense_embedding_model

        # If FastEmbed dense model enabled
        if isinstance(self._dense_embedding_model,str):
            self._client.set_model(embedding_model_name = self._dense_embedding_model,
                                   cache_dir = embedding_folder_cached)

    def __create_collection(self,
                            collection_name :str,
                            vectors_config: Union[VectorParams,dict],
                            shard_number :int = 2,
                            quantization_mode :Literal['binary','scalar','product','none'] = "scalar",
                            default_segment_number :int = 4,
                            always_ram :bool = True) -> None:
        """Create collection with default name
        Parameters:
        - collection_name (str): The name of desired collection
        - shard_number (int): The number of parallel processes as the same time. Default is 2,
        - quantization_mode (Literal): If enabled, it brings more compact representation embedding,then cache
        more in RAM and reduce the number of disk reads. With scalar, compression with be up to 4x times
        (float32 -> uint8) with the most balance in accuracy and speed. Binary is extreme case of scalar, reducing the
        memory footprint by 32 (with limited model), and the most rapid mode. Product is the slower method, and loss of
        accuracy, only recommended for high dimensional vectors.
        - default_segment_number (int). Default is 4. Larger value will enhance the latency, smaller one the throughput.
        - always_ram (bool): Default is True, indicated that quantized vectors is persisted on RAM
        """
        assert collection_name, "Collection name must be a string"

        # Whe collection is existed or not
        if not self._client.collection_exists(collection_name):
            # Default is None
            quantization_config = None
            # Define quantization mode if enable
            if quantization_mode == "scalar":
                # Scalar mode, currently Qdrant only support INT8
                quantization_config = models.ScalarQuantization(
                    scalar = models.ScalarQuantizationConfig(
                        type = models.ScalarType.INT8,
                        quantile = 0.99, # if specify 0.99, 1% of extreme values will be excluded from the quantization bounds.
                        always_ram = always_ram
                    )
                )
            elif quantization_mode == "binary":
                # Binary mode
                quantization_config = models.BinaryQuantization(
                    binary = models.BinaryQuantizationConfig(
                        always_ram = always_ram,
                    ),
                ),
            elif quantization_mode == "product":
                # Product quantization mode
                quantization_config = models.ProductQuantization(
                    product = models.ProductQuantizationConfig(
                        compression = models.CompressionRatio.X16, # Default X16
                        always_ram = always_ram,
                    ),
                ),

            # Optimizer config
            # When indexing threshold is 0, It will enable to avoid unnecessary indexing of vectors,
            # which will be overwritten by the next batch.
            optimizers_config = models.OptimizersConfigDiff(default_segment_number = default_segment_number,
                                                            indexing_threshold = 0)
            # Create collection
            self._client.create_collection(
                collection_name = collection_name,
                vectors_config = vectors_config,
                shard_number = shard_number,
                quantization_config = quantization_config,
                optimizers_config = optimizers_config
            )
            # Update collection
            self._client.update_collection(
                collection_name = collection_name,
                optimizer_config = models.OptimizersConfigDiff(indexing_threshold = 20000),
            )


    def __get_embeddings(self,
                         texts :list[str],
                         embedding_model : BaseEmbedding,
                         batch_size :int,
                         num_workers :int,
                         show_progress :bool = True) -> List[Embedding]:
        """Return embedding from documents"""
        # Set batch size and num workers
        embedding_model.num_workers = num_workers
        embedding_model.embed_batch_size = batch_size
        # Other information
        model_infor = embedding_model.dict()
        callback_manager = embedding_model.callback_manager
        # Return embedding
        return embedding_model.get_text_embedding_batch(texts = texts, show_progress = show_progress)

    def __convert_documents_to_payloads(self,
                                   documents :Sequence[BaseNode],
                                   embedding_model_name :Optional[str] = None,
                                   include_embedding_name :bool = True) -> list[dict]:
        """Construct the payload data from LlamaIndex document/node datatype
        Parameter:
        - documents (BaseNode): The list of BaseNode datatype in LlamaIndex
        - embedding_model_name (str): The name of the embedding model (For adding payloads information)
        - include_embedding_name (bool): Specify whether adding title or not"""
        # Clear private data from payload
        for i in range(len(documents)):
            documents[i].embedding = None
            # Pop file path
            documents[i].metadata.pop("file_path")
            # documents[i].excluded_embed_metadata_keys = []
            # documents[i].excluded_llm_metadata_keys = []
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
        if include_embedding_name and embedding_model_name != None:
            for i in range(len(payloads)): payloads[i].update({"embedding_model_name": embedding_model_name})
        return payloads

    def __convert_score_point_to_node_with_score(self,scored_points :List[ScoredPoint]) -> Sequence[NodeWithScore]:
        text_nodes = [TextNode.from_dict(point.payload["_node_content"]) for point in scored_points]
        # return NodeWithScore
        return [NodeWithScore(node = text_nodes[i], score = point.score) for (i,point) in enumerate(scored_points)]

    def __convert_query_response_to_node_with_score(self,scored_points :List[QueryResponse]) -> Sequence[NodeWithScore]:
        text_nodes = [TextNode.from_dict(point.metadata["_node_content"]) for point in scored_points]
        # return NodeWithScore
        return [NodeWithScore(node = text_nodes[i], score = point.score) for (i,point) in enumerate(scored_points)]

    def __insert_points(self,
                        list_embeddings :list[list[float]],
                        list_payloads :list[dict],
                        point_ids: Optional[list[str]] = None,
                        collection_name :Optional[str] = None,
                        batch_size :int = _DEFAULT_UPLOAD_BATCH_SIZE,
                        parallel :int = 1) -> None:
        """Insert point to the collection"""
        # Check size
        if not len(list_embeddings) == len(list_payloads):
            raise Exception("Number of embeddings must be equal with number of payloads")
        if point_ids == None:
            point_ids = [str(uuid4()) for i in range(len(list_embeddings))]

        # Collection name
        collection_name = collection_name if collection_name != None else self._collection_name
        # Upload point
        self._client.upload_collection(collection_name = collection_name,
                                       ids = point_ids,
                                       vectors = list_embeddings,
                                       payload = list_payloads,
                                       batch_size = batch_size,
                                       parallel = parallel)

    def insert_documents(self,
                         documents :Sequence[BaseNode],
                         embedded_batch_size: int = 64,
                         embedded_num_workers: Optional[int] = None) -> None:
        # Get content and its embedding
        contents = [doc.get_content() for doc in documents]
        embeddings = None

        # Define dense embedding
        if isinstance(self._dense_embedding_model, BaseEmbedding):
            # With LlamaIndex Embedding
            # Get embedding model name
            model_name = self._dense_embedding_model.model_name
            # Define embedding
            embeddings = self.__get_embeddings(texts = contents,
                                               embedding_model = self._dense_embedding_model,
                                               batch_size = embedded_batch_size,
                                               num_workers = embedded_num_workers)

            # Get embedding dimension
            embedding_dimension = len(embeddings[0])
            # Define vector config
            vectors_config = VectorParams(size = embedding_dimension,
                                          distance = self._distance,
                                          on_disk = self._on_disk,
                                          hnsw_config = models.HnswConfigDiff(on_disk = self._on_disk))
        else:
            # When embedding model is str, default is activated with FastEmbed model
            vectors_config = self._client.get_fastembed_vector_params()
            # Get params
            model_name = list(vectors_config)[0]

        # Define payloads
        payloads = self.__convert_documents_to_payloads(documents = documents,
                                                        embedding_model_name = model_name)
        # Create collection if doesn't exist!
        self.__create_collection(collection_name = self._collection_name,
                                 vectors_config = vectors_config)
        # Insert vector to collection
        if isinstance(self._dense_embedding_model, BaseEmbedding):
            # With BaseEmbedding model
            self.__insert_points(list_embeddings = embeddings, list_payloads = payloads)
        else:
            # With FastEmbed Model
            self._client.add(collection_name = self._collection_name,
                             documents = contents,
                             metadata = payloads)

    def reembedding_with_collection(self,
                                    embedding_model : BaseEmbedding,
                                    collection_name: Optional[str] = None,
                                    upload_batch_size :int = _DEFAULT_UPLOAD_BATCH_SIZE,
                                    embedded_batch_size :int = 64,
                                    embedded_num_workers :int = 4,
                                    show_progress :bool = True) -> None:
        """Re-embedding the existed collection to another collection"""
        # Get points from current collection
        points,_ = self._get_points()

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
    def __search(self,
                 query_vector: List[Num],
                 filter : Optional[models.Filter] = None,
                 similarity_top_k :int = 3,
                 rescore :bool = True) -> List[types.ScoredPoint]:
        """Search and return top-k result from input embedding vector
        Parameter:
        - query_vector (List[Num]): List of value represent for sematic embedding of query.
        filter (Filter): Filter the result under conditions.
        similarity_top_k (int): Determine the number of result should be returned.
        rescore (bool): Disable rescoring, which will reduce the number of disk reads, but slightly decrease the
        precision"""
        if not self._client.collection_exists(self._collection_name):
            raise Exception(f"Collection {self._collection_name} isn't existed!")

        # Search params
        # search_params = models.SearchParams(hnsw_ef=512, exact=False)
        search_params = None
        # Disable rescore method
        if not rescore:
            search_params = models.SearchParams(
                quantization=models.QuantizationSearchParams(rescore = False)
            )

        # Return search
        return self._client.search(
            collection_name = self._collection_name,
            query_vector = query_vector,
            limit = similarity_top_k,
            search_params = search_params,
            query_filter = filter,
        )

    def retrieve(self,
                 query :str,
                 similarity_top_k :int = 3) -> Sequence[NodeWithScore]:
        """Retrieve nodes from vector store corresponding to question.
        :param
        - query (str): The query str for retrieve.
        - similarity_top_k (int). Default is 3. Return top-k element from retrieval"""

        # With LlamaIndex Embedding case
        if isinstance(self._dense_embedding_model, BaseEmbedding):
            # Get query embedding
            query_embedding = self._dense_embedding_model.get_query_embedding(query = query)
            # Get nodes
            scored_points = self.__search(query_vector = query_embedding,
                                          similarity_top_k = similarity_top_k)
            # Convert to node with score
            return self.__convert_score_point_to_node_with_score(scored_points=scored_points)
        else:
            # With FastEmbed model
            # Get nodes
            scored_points = self._client.query(query_text = query,
                                               collection_name = self._collection_name,
                                               limit = similarity_top_k)
            # Convert to node with score
            return self.__convert_query_response_to_node_with_score(scored_points = scored_points)

    def update_point(self, id, vector):
        result = self._client.update_vectors(
            collection_name = self._collection_name,
            points = [
                models.PointVectors(
                    id = id,
                    vector = vector
                )])
        print(result)

    def _retrieve_points(self, ids :list[Union[str,int]]):
        return self._client.retrieve(collection_name = self._collection_name,
                                     ids = ids,
                                     with_vectors = True)
    def _collection_info(self) -> types.CollectionInfo:
        return self._client.get_collection(self._collection_name)

    def _count_points(self) -> int:
        # Get total amount of points
        result = self._client.count(self._collection_name)
        return result.count

    def _get_points(self,
                   limit :Optional[int] = "all",
                   with_vector :bool = False) -> Tuple[List[types.Record], Optional[types.PointId]]:
        """Get all the point in the Qdrant collection or with limited amount"""
        # Get total point
        total_points = self._count_points()

        # Limit if specify
        if limit == "all": limit = total_points
        # Return point
        return self._client.scroll(collection_name = self._collection_name,
                                   limit = limit,
                                   with_vectors = with_vector)

    def __set_payload(self, point :list[Union[str,int]]):
        self._client.set_payload(collection_name = self._collection_name,
                                 payload = {},
                                 points = point)