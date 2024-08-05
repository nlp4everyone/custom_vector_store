from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, ScoredPoint, QueryResponse, Filter
from qdrant_client.conversions import common_types as types
from llama_index.core.schema import BaseNode, NodeWithScore, TextNode
from llama_index.core.base.embeddings.base import BaseEmbedding
from typing import Optional, Union, List, Tuple, Sequence, Literal
from fastembed import (TextEmbedding,
                       SparseTextEmbedding)
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
                 enable_hybrid :bool = False,
                 enable_semantic_cache: bool = False,
                 dense_embedding_model: Union[BaseEmbedding, str] = "BAAI/bge-base-en-v1.5",
                 spare_embedding_model: Optional[str] = "prithvida/Splade_PP_en_v1",
                 distance: Distance = Distance.COSINE,
                 embedding_folder_cached: str = "cached",
                 semantic_cache_threshold: float = 0.4,
                 shard_number: int = 2,
                 quantization_mode: Literal['binary', 'scalar', 'product', 'none'] = "scalar",
                 default_segment_number: int = 4,
                 on_disk: bool = True) -> None:

        """
        Init Qdrant client service

        :param collection_name: The name of collection (Required).
        :type collection_name: str
        :param url: The Qdrant url string.
        :type url: str
        :param port: Qdrant port. Default is 6333.
        :type port: int
        :param grpc_port: Grpc port. Default is 6334.
        :type grpc_port: int
        :param prefer_grpc: Whether prefer grpc or not
        :type prefer_grpc: bool
        :param api_key: api key for connecting
        :type api_key: str
        :param enable_hybrid: Enable hybrid search. Default is False.
        :type enable_hybrid: bool
        :param enable_semantic_cache: Enable semantic cache optimization. Default is False.
        :type enable_semantic_cache: bool
        :param dense_embedding_model: The dense embedding model. Default is BAAI/bge-base-en-v1.5
        :type dense_embedding_model: str
        :param spare_embedding_model: The dense embedding model. Default is prithvida/Splade_PP_en_v1.
        :type spare_embedding_model: str
        :param distance: The calculated distance for similarity search. Default is Cosine.
        :type distance: Distance
        :param embedding_folder_cached: Directory path for saving model.
        :type embedding_folder_cached: str
        :param semantic_cache_threshold: Threshold for semantic cache. Default is 0.45
        :type semantic_cache_threshold: float
        :param shard_number: The number of parallel processes as the same time. Default is 2.
        :type shard_number: int
        :param quantization_mode: Include scalar, binary and product.
        :type quantization_mode: Literal
        :param default_segment_number: Default is 4. Larger value will enhance the latency, smaller one the throughput.
        :type default_segment_number: int
        """
        assert collection_name, "Collection name must be string"
        self._client = QdrantClient(url = url,
                                    port = port,
                                    grpc_port = grpc_port,
                                    api_key = api_key,
                                    prefer_grpc = prefer_grpc)
        # Set value
        self.collection_name = collection_name
        self.enable_hybrid = enable_hybrid
        self.enable_semantic_cache = enable_semantic_cache
        self.on_disk = on_disk
        self.distance = distance
        self.dense_embedding_model = dense_embedding_model
        self.shard_number = shard_number
        self.quantization_mode = quantization_mode
        self.default_segment_number = default_segment_number
        self.cache_collection_name = f"cache_{self.collection_name}"
        self.semantic_cache_threshold = semantic_cache_threshold
        self.spare_embedding_model = spare_embedding_model
        self.embedding_folder_cached = embedding_folder_cached

        # Set embed model
        self._set_embed_model()
        # Set hybrid mode
        self._set_hybrid_mode(enable = self.enable_hybrid)

    def _set_hybrid_mode(self, enable :bool = False):
        """Set hybrid mode if enabled"""
        # Enable hybrid search
        if enable:
            # Get list supported models
            sparse_supported_models = SparseTextEmbedding.list_supported_models()
            sparse_supported_models = [model['model'] for model in sparse_supported_models]

            if isinstance(self.spare_embedding_model, str):
                # Check Dense EmbedModel is available
                if self.spare_embedding_model not in sparse_supported_models:
                    raise Exception(f"{self.spare_embedding_model} is not supported!")
            # Set model
            self._client.set_sparse_model(embedding_model_name = self.spare_embedding_model,
                                           cache_dir = self.embedding_folder_cached)

    def _set_embed_model(self):
        """Set local embedding model (FastEmbed) if enabled"""
        # If FastEmbed dense model enabled
        if isinstance(self.dense_embedding_model, str):
            # Get list supported models
            dense_supported_models = TextEmbedding.list_supported_models()
            dense_supported_models = [model['model'] for model in dense_supported_models]

            # Check Dense EmbedModel is available
            if self.dense_embedding_model not in dense_supported_models:
                raise Exception(f"{self.dense_embedding_model} is not supported!")
            # Set model
            self._client.set_model(embedding_model_name = self.dense_embedding_model,
                                    cache_dir = self.embedding_folder_cached)

    def _set_cache_collection(self):
        """Enable when set semantic cache search"""
        # Define cache collection name
        collection_info = self._collection_info(collection_name = self.collection_name)

        # Check collection exists
        status = self._client.collection_exists(self.cache_collection_name)

        # Sparse vector
        sparse_vector_config = None
        if self.enable_hybrid:
            sparse_vector_config = self._client.get_fastembed_sparse_vector_params()

        # When cache collection is not exists
        if not status:
            # Get vector config
            vector_config = collection_info.config.params.vectors
            # Create cache collection from main
            self._client.create_collection(collection_name = self.cache_collection_name,
                                            vectors_config = vector_config,
                                            sparse_vectors_config = sparse_vector_config)
            # May need to enhance quality

    def _semantic_cache_search(self,
                               query :str,
                               semantic_cache_threshold: float,
                               similarity_top_k :int = 3,
                               cache_similarity_top_k :int = 3,
                               filter :Optional[Filter] = None) -> Sequence[NodeWithScore]:
        """
        Semantic Cache Search flows
        :param query: The search query (Required)
        :type query: str
        :param semantic_cache_threshold: The threshold for retrieving from cache collection.
        :type semantic_cache_threshold: float
        :param similarity_top_k: Top k result from similarity search with base collection
        :type similarity_top_k: int
        :param cache_similarity_top_k: Top k result from similarity search with cache collection
        :type cache_similarity_top_k: int
        :param filter: Conditional filter
        :type filter: Filter
        :return: Return a sequence of NodeWithScore
        :rtype: Sequence
        """

        # Count cache point
        cache_points = self._count_points(collection_name = self.cache_collection_name)
        # If zero points in cached collection
        if cache_points == 0:
            nodes = self.__query(collection_name = self.collection_name,
                                 query = query,
                                 similarity_top_k = similarity_top_k,
                                 filter = filter,
                                 return_type = "NodeWithScore")
            # Convert NodeWithScore to TextNode
            base_nodes = [node.node for node in nodes]
            # Add point to cache collection
            self.insert_documents(documents = base_nodes,
                                  collection_name = self.cache_collection_name)
            return nodes
        else:
            # Search with cache collection
            cache_nodes = self.__query(collection_name = self.cache_collection_name,
                                       query = query,
                                       similarity_top_k = cache_similarity_top_k,
                                       filter = filter,
                                       score_threshold = semantic_cache_threshold,
                                       return_type = "NodeWithScore")
            # When find out nodes satisfied threshold condition.
            if len(cache_nodes) > 0:
                return cache_nodes

            # When no nodes in cache collection achieved!
            nodes = self.__query(collection_name = self.collection_name,
                                 query = query,
                                 similarity_top_k = similarity_top_k,
                                 filter = filter,
                                 return_type = "NodeWithScore")

            # Convert NodeWithScore to TextNode
            base_nodes = [node.node for node in nodes]
            # Add node to cache collection (Only add non duplicated nodes)
            self.insert_documents(documents = base_nodes,
                                  collection_name = self.cache_collection_name)
            return nodes

    def __create_collection(self,
                            collection_name :str,
                            dense_vectors_config: Union[VectorParams,dict],
                            sparse_vectors_config :Optional[str] = None,
                            shard_number :int = 2,
                            quantization_mode :Literal['binary','scalar','product','none'] = "scalar",
                            default_segment_number :int = 4,
                            always_ram :bool = True) -> None:
        """
        Create collection with default name

        Args:
            collection_name: The name of desired collection
            shard_number: The number of parallel processes as the same time. Default is 2.
            quantization_mode: Quantization mode.
            default_segment_number: Default is 4. Larger value will enhance the latency, smaller one the throughput.
            always_ram: Indicated that quantized vectors is persisted on RAM.
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
                vectors_config = dense_vectors_config,
                sparse_vectors_config = sparse_vectors_config,
                shard_number = shard_number,
                quantization_config = quantization_config,
                optimizers_config = optimizers_config
            )
            # Update collection
            self._client.update_collection(
                collection_name = collection_name,
                optimizer_config = models.OptimizersConfigDiff(indexing_threshold = 20000),
            )



    def __insert_points(self,
                        list_embeddings :list[list[float]],
                        list_payloads :list[dict],
                        point_ids: Optional[list[str]] = None,
                        collection_name :Optional[str] = None,
                        batch_size :int = _DEFAULT_UPLOAD_BATCH_SIZE,
                        parallel :int = 1) -> None:
        """
        Insert point to the collection

        Args:
            list_embeddings (Required): List of embeddings
            list_payloads (List[dict]) (Required): List of payloads
            point_ids (list[str]) (Optional): List of point id
            collection_name (str) (Optional): Name of collection for importing
            batch_size (int): The desired batch size
            parallel (int): The desired batch size
        """

        # Check size
        if not len(list_embeddings) == len(list_payloads):
            raise Exception("Number of embeddings must be equal with number of payloads")
        if point_ids == None:
            point_ids = [str(uuid4()) for i in range(len(list_embeddings))]

        # Collection name
        if collection_name == None: collection_name = self.collection_name

        # Upload point
        self._client.upload_collection(collection_name = collection_name,
                                       ids = point_ids,
                                       vectors = list_embeddings,
                                       payload = list_payloads,
                                       batch_size = batch_size,
                                       parallel = parallel)

    def insert_documents(self,
                         documents :Sequence[BaseNode],
                         collection_name: Optional[str] = None,
                         embedded_batch_size: int = 64,
                         embedded_num_workers: Optional[int] = None,
                         upload_batch_size: int = 16,
                         upload_parallel :Optional[int] = None) -> None:
        """
        Insert document to collection.

        :param documents: List of BaseNode.
        :type documents: Sequence[BaseNode]
        :param embedded_batch_size: Batch size for embedding model. Default is 64.
        :type embedded_batch_size: int
        :param embedded_num_workers: Batch size for embedding model (Optional). Default is None.
        :type embedded_num_workers: int
        :param upload_batch_size: Batch size for uploading points. Default is 16.
        :type upload_batch_size: int
        :param upload_parallel: Number of parallel for uploading point (Optional). Default is None.
        :type upload_parallel: Optional[int]
        """

        # Get collection name
        if collection_name == None: collection_name = self.collection_name

        # Get content and its embedding
        contents = [doc.get_content() for doc in documents]
        embeddings = None

        # Define dense embedding
        if isinstance(self.dense_embedding_model, BaseEmbedding):
            # With LlamaIndex Embedding
            # Get embedding model name
            model_name = self.dense_embedding_model.model_name
            # Define embedding
            embeddings = self.__get_embeddings(texts = contents,
                                               embedding_model = self.dense_embedding_model,
                                               batch_size = embedded_batch_size,
                                               num_workers = embedded_num_workers)

            # Get embedding dimension
            embedding_dimension = len(embeddings[0])
            # Define vector config
            dense_vectors_config = VectorParams(size = embedding_dimension,
                                                distance = self.distance,
                                                on_disk = self.on_disk,
                                                hnsw_config = models.HnswConfigDiff(on_disk = self.on_disk))
        else:
            # When embedding model is str, default is activated with FastEmbed model
            dense_vectors_config = self._client.get_fastembed_vector_params()
            # Get params
            model_name = list(dense_vectors_config)[0]

        sparse_embedding_model = None
        # Hybrid Search enbable
        if self.enable_hybrid:
            sparse_embedding_model = self._client.get_fastembed_sparse_vector_params()

        # Define payloads
        payloads = self.convert_documents_to_payloads(documents = documents,
                                                       embedding_model_name = model_name)
        # Create collection if doesn't exist!
        self.__create_collection(collection_name = collection_name,
                                 dense_vectors_config = dense_vectors_config,
                                 sparse_vectors_config = sparse_embedding_model,
                                 shard_number = self.shard_number,
                                 quantization_mode = self.quantization_mode,
                                 default_segment_number = self.default_segment_number)

        # Create cache collection, if enabled:
        if self.enable_semantic_cache:
            # Set semantic cache
            self._set_cache_collection()

        # Insert vector to collection
        if isinstance(self.dense_embedding_model, BaseEmbedding):
            # With BaseEmbedding model
            self.__insert_points(list_embeddings = embeddings,
                                 list_payloads = payloads,
                                 batch_size = upload_batch_size,
                                 parallel = upload_parallel)
        else:
            # With FastEmbed Model
            self._client.add(collection_name = self.collection_name,
                             documents = contents,
                             metadata = payloads,
                             batch_size = upload_batch_size,
                             parallel = upload_parallel)

    def __query(self,
                collection_name :str,
                query: str,
                similarity_top_k: int = 3,
                filter: Optional[Filter] = None,
                score_threshold :Optional[float] = None,
                rescore :bool = True,
                return_type :Literal["NodeWithScore","default"] = "NodeWithScore"
                ) -> Union[Sequence[NodeWithScore],List[models.ScoredPoint],List[models.QueryResponse]]:
        """
        Retrieve nodes from vector store corresponding to question.

        :param query: The query str for retrieve (Required)
        :type query: str
        :param similarity_top_k: Default is 3. Return top-k element from retrieval.
        :type similarity_top_k: int
        :param filter: Conditional filter for searching. Default is None
        :type filter: Filter
        :return: Return a sequence of NodeWithScore
        :rtype Sequence[NodeWithScore]
        """

        # With LlamaIndex Embedding case
        if isinstance(self.dense_embedding_model, BaseEmbedding):
            # Get query embedding
            query_embedding = self.dense_embedding_model.get_query_embedding(query = query)

            search_params = None
            # Disable rescore method
            if not rescore:
                search_params = models.SearchParams(
                    quantization = models.QuantizationSearchParams(rescore=False)
                )
            # Return search
            scored_points = self._client.search(collection_name = collection_name,
                                                query_vector = query_embedding,limit = similarity_top_k,
                                                search_params = search_params,
                                                query_filter = filter,
                                                score_threshold = score_threshold)
            # Convert to node with score
            return self.convert_score_point_to_node_with_score(scored_points = scored_points) if return_type == "NodeWithScore" else scored_points

        else:
            # With FastEmbed model
            # Get nodes
            scored_points = self._client.query(query_text = query,
                                               collection_name = collection_name,
                                               query_filter = filter,
                                               limit = similarity_top_k,
                                               score_threshold = score_threshold)
            # Convert to node with score
            return self.convert_query_response_to_node_with_score(scored_points = scored_points) if return_type == "NodeWithScore" else scored_points

    def retrieve(self,
                 query: str,
                 similarity_top_k: int = 3,
                 filter: Optional[Filter] = None) -> Sequence[NodeWithScore]:
        """
        Retrieve nodes from vector store corresponding to question.

        :param query: The query str for retrieve (Required)
        :type query: str
        :param similarity_top_k: Default is 3. Return top-k element from retrieval.
        :type similarity_top_k: int
        :param filter: Conditional filter for searching. Default is None
        :type filter: Filter
        :return: Return a sequence of NodeWithScore
        :rtype Sequence[NodeWithScore]
        """

        # Check base collection
        status = self._client.collection_exists(collection_name = self.collection_name)
        if not status:
            raise Exception(f"Collection {self.collection_name} isn't existed")
        # Check collection
        count_points = self._count_points(collection_name = self.collection_name)
        if count_points == 0:
            raise Exception(f"Collection {self.collection_name} is empty!")
        # Create cache collection
        self._set_cache_collection()

        # If semantic cache enabled
        if self.enable_semantic_cache:
            # Check cache collection
            status = self._client.collection_exists(collection_name=self.cache_collection_name)
            if not status:
                raise Exception(f"Collection {self.cache_collection_name} not existed")

            # Enable semantic cache search
            nodes = self._semantic_cache_search(query = query,
                                                semantic_cache_threshold = self.semantic_cache_threshold,
                                                similarity_top_k = similarity_top_k,
                                                cache_similarity_top_k = similarity_top_k,
                                                filter = filter)
            return nodes

        else:
            # Enable base search
            nodes = self.__query(collection_name = self.collection_name,
                                 query = query,
                                 similarity_top_k = similarity_top_k,
                                 filter = filter,
                                 return_type = "NodeWithScore")
            return nodes

    def update_point(self, id, vector):
        """Update value for points"""
        result = self._client.update_vectors(
            collection_name = self.collection_name,
            points = [
                models.PointVectors(
                    id = id,
                    vector = vector
                )])
        print(result)

    def _retrieve_points(self, ids :list[Union[str,int]]):
        """
        Retrieve point by specifying ids

        Args:
            ids (list(Union[str,int]): The list ids of desired points.
        """
        return self._client.retrieve(collection_name = self.collection_name,
                                     ids = ids,
                                     with_vectors = True)
    def _collection_info(self, collection_name: str) -> types.CollectionInfo:
        """Return collection info"""
        # Check collection exist
        if not self._client.collection_exists(collection_name):
            raise Exception(f"Collection {collection_name} is not exist!")
        # Return information
        return self._client.get_collection(collection_name)

    def _count_points(self, collection_name: str) -> int:
        """Return the total amount of point inside collection"""
        # Check collection exist
        status = self._client.collection_exists(collection_name)
        if not status:
            raise Exception(f"Collection {collection_name} is not exist!")

        # Get total amount of points
        result = self._client.count(self.collection_name)
        return result.count

    @staticmethod
    def __get_embeddings(texts: list[str],
                         embedding_model: BaseEmbedding,
                         batch_size: int,
                         num_workers: int,
                         show_progress: bool = True) -> List[Embedding]:
        """
        Return embedding from documents

        Args:
            texts (list[str]): List of input text
            embedding_model (BaseEmbedding): The text embedding model
            batch_size (int): The desired batch size
            num_workers (int): The desired num workers
            show_progress (bool): Indicate show progress or not

        Returns:
             Return list of Embedding
        """
        # Set batch size and num workers
        embedding_model.num_workers = num_workers
        embedding_model.embed_batch_size = batch_size
        # Other information
        model_infor = embedding_model.dict()
        callback_manager = embedding_model.callback_manager
        # Return embedding
        return embedding_model.get_text_embedding_batch(texts=texts, show_progress=show_progress)

    @staticmethod
    def convert_documents_to_payloads(documents: Sequence[BaseNode],
                                      embedding_model_name: Optional[str] = None,
                                      include_embedding_name: bool = True) -> list[dict]:
        """
        Construct the payload data from LlamaIndex document/node datatype

        Args:
            documents (BaseNode): The list of BaseNode datatype in LlamaIndex
            embedding_model_name (str): The name of the embedding model (For adding payloads information)
            include_embedding_name (bool): Specify whether adding title or not

        Returns:
            Payloads (list[dict).
        """

        # Clear private data from payload
        for i in range(len(documents)):
            documents[i].embedding = None
            # Pop file path
            documents[i].metadata["file_path"] = "",
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
                     "ref_doc_id": document.id_} for document in documents]

        # Include embedding name if specify
        if include_embedding_name and embedding_model_name != None:
            for i in range(len(payloads)): payloads[i].update({"embedding_model_name": embedding_model_name})
        return payloads

    @staticmethod
    def convert_score_point_to_node_with_score(scored_points: List[ScoredPoint]) -> Sequence[NodeWithScore]:
        """
        Convert ScorePoint Datatype (Qdrant) to NodeWithScore Datatype (LlamaIndex)

        Args:
            scored_points (List[ScoredPoint]): List of ScoredPoint
        Returns:
            Sequence of NodeWithScore
        """

        # Define text nodes
        text_nodes = [TextNode.from_dict(point.payload["_node_content"]) for point in scored_points]
        # return NodeWithScore
        return [NodeWithScore(node=text_nodes[i], score=point.score) for (i, point) in enumerate(scored_points)]

    @staticmethod
    def convert_query_response_to_node_with_score(scored_points: List[QueryResponse]) -> Sequence[NodeWithScore]:
        """
        Convert QueryResponse Datatype (Qdrant) to NodeWithScore Datatype (LlamaIndex)

        Args:
            scored_points (List[QueryResponse]): List of QueryResponse
        Returns:
            Sequence of NodeWithScore"""

        # Define text nodes
        text_nodes = [TextNode.from_dict(point.metadata["_node_content"]) for point in scored_points]
        # return NodeWithScore
        return [NodeWithScore(node=text_nodes[i], score=point.score) for (i, point) in enumerate(scored_points)]

    def _get_points(self,
                    collection_name :str,
                    limit :Optional[int] = "all",
                    with_vector :bool = False) -> Tuple[List[types.Record], Optional[types.PointId]]:
        """
        Get all the point in the Qdrant collection or with limited amount

        Args:
            limit (int, optional): The number of point retrieved. Default is all.
            with_vector (bool): Whether return vector or not.
        """
        # Get total point
        total_points = self._count_points(collection_name = collection_name)

        # Limit if specify
        if limit == "all": limit = total_points
        # Return point
        return self._client.scroll(collection_name = collection_name,
                                   limit = limit,
                                   with_vectors = with_vector)

    def __set_payload(self, point :list[Union[str,int]]):
        """Set payload"""
        self._client.set_payload(collection_name = self.collection_name,
                                 payload = {},
                                 points = point)

    def reembedding_with_collection(self,
                                    embedding_model : BaseEmbedding,
                                    collection_name: Optional[str] = None,
                                    upload_batch_size :int = _DEFAULT_UPLOAD_BATCH_SIZE,
                                    embedded_batch_size :int = 64,
                                    embedded_num_workers :int = 4,
                                    show_progress :bool = True) -> None:
        """Re-embedding the existed collection to another collection"""
        # Get points from current collection
        points,_ = self._get_points(collection_name = collection_name)

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
            collection_name = f"{self.collection_name}_{id}"

        # Get embedding dimension
        embedding_dimension = len(embeddings[0])
        # Define vector config
        dense_vectors_config = VectorParams(size = embedding_dimension,
                                            distance = self.distance,
                                            on_disk = self.on_disk,
                                            hnsw_config = models.HnswConfigDiff(on_disk=self.on_disk))
        # Create collection if doesn't exist!
        self.__create_collection(collection_name = collection_name,
                                 dense_vectors_config = dense_vectors_config)

        # Insert
        self.__insert_points(collection_name = collection_name,
                             list_embeddings = embeddings,
                             list_payloads = payloads,
                             point_ids = points_ids,
                             batch_size = upload_batch_size)