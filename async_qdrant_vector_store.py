from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import Distance, VectorParams
from qdrant_client.conversions import common_types as types
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.core.base.embeddings.base import BaseEmbedding
from typing import Optional, Union, List, Tuple, Sequence, Literal
from qdrant_vector_store import QdrantVectorStore
from fastembed import (TextEmbedding,
                       SparseTextEmbedding)
from uuid import uuid4

# DataType
Num = Union[int, float]
Embedding = List[float]

# Params
_DEFAULT_UPLOAD_BATCH_SIZE = 64


class AsyncQdrantVectorStore(QdrantVectorStore):
    def __init__(self,
                 collection_name :str,
                 url :str = "http://localhost:6333",
                 port :int = 6333,
                 grpc_port :int = 6334,
                 prefer_grpc :bool = False,
                 api_key :Optional[str] = None,
                 hybrid_search :bool = False,
                 dense_embedding_model: Union[BaseEmbedding, str] = "BAAI/bge-base-en-v1.5",
                 spare_embedding_model: Optional[str] = "prithvida/Splade_PP_en_v1",
                 distance: Distance = Distance.COSINE,
                 embedding_folder_cached: str = "cached",
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
        :param hybrid_search: Enable hybrid search. Default is False.
        :type hybrid_search: bool
        :param dense_embedding_model: The dense embedding model. Default is BAAI/bge-base-en-v1.5
        :type dense_embedding_model: str
        :param spare_embedding_model: The dense embedding model. Default is prithvida/Splade_PP_en_v1.
        :type spare_embedding_model: str
        :param distance: The calculated distance for similarity search. Default is Cosine.
        :type distance: Distance
        :param embedding_folder_cached: Directory path for saving model.
        :type embedding_folder_cached: str
        :param shard_number: The number of parallel processes as the same time. Default is 2.
        :type shard_number: int
        :param quantization_mode: Include scalar, binary and product.
        :type quantization_mode: Literal
        :param default_segment_number: Default is 4. Larger value will enhance the latency, smaller one the throughput.
        :type default_segment_number: int
        """

        # Set value
        self._collection_name = collection_name
        self._hybrid_search = hybrid_search
        self._on_disk = on_disk
        self._distance = distance
        self._dense_embedding_model = dense_embedding_model
        self._shard_number = shard_number
        self._quantization_mode = quantization_mode
        self._default_segment_number = default_segment_number

        # Inherit
        super().__init__(collection_name = collection_name,
                         url = url,
                         port = port,
                         grpc_port = grpc_port,
                         prefer_grpc = prefer_grpc,
                         api_key = api_key,
                         hybrid_search = hybrid_search,
                         dense_embedding_model = dense_embedding_model,
                         spare_embedding_model = spare_embedding_model,
                         distance = distance,
                         embedding_folder_cached = embedding_folder_cached,
                         shard_number = shard_number,
                         quantization_mode = quantization_mode,
                         default_segment_number = default_segment_number,
                         on_disk = on_disk)

        # assert collection_name, "Collection name must be string"
        self._client = AsyncQdrantClient(url = url,
                                         port = port,
                                         grpc_port = grpc_port,
                                         api_key = api_key,
                                         prefer_grpc = prefer_grpc)
        # Set embed model
        self._set_embed_model()
        # Set hybrid mode
        self._set_hybrid_mode()

    async def __create_collection(self,
                                  collection_name :str,
                                  dense_vectors_config: Union[VectorParams,dict],
                                  sparse_vectors_config :Optional[str] = None,
                                  shard_number :int = 2,
                                  quantization_mode :Literal['binary','scalar','product','none'] = "scalar",
                                  default_segment_number :int = 4,
                                  always_ram :bool = True) -> None:
        """
        Create collection with default name

        :param collection_name: The name of desired collection
        :type collection_name: str
        :param shard_number: The number of parallel processes as the same time. Default is 2.
        :type shard_number: int
        :param quantization_mode: If enabled, it brings more compact representation embedding,then cache
        more in RAM and reduce the number of disk reads. With scalar, compression with be up to 4x times
        (float32 -> uint8) with the most balance in accuracy and speed. Binary is extreme case of scalar, reducing the
        memory footprint by 32 (with limited model), and the most rapid mode. Product is the slower method, and loss of
        accuracy, only recommended for high dimensional vectors.
        :type quantization_mode: Literal
        :param default_segment_number: Default is 4. Larger value will enhance the latency, smaller one the throughput.
        :type: int
        :param always_ram: Default is True, indicated that quantized vectors is persisted on RAM.
        :type always_ram: bool
        """
        assert collection_name, "Collection name must be a string"

        # Whe collection is existed or not
        collection_status = await self._client.collection_exists(collection_name)
        if not collection_status:
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
            await self._client.create_collection(
                collection_name = collection_name,
                vectors_config = dense_vectors_config,
                sparse_vectors_config = sparse_vectors_config,
                shard_number = shard_number,
                quantization_config = quantization_config,
                optimizers_config = optimizers_config
            )
            # Update collection
            await self._client.update_collection(
                collection_name = collection_name,
                optimizer_config = models.OptimizersConfigDiff(indexing_threshold = 20000),
            )

    async def __get_embeddings(self,
                               texts :list[str],
                               embedding_model : BaseEmbedding,
                               batch_size :int,
                               num_workers :int,
                               show_progress :bool = True) -> List[Embedding]:
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
        embeddings = await embedding_model.aget_text_embedding_batch(texts = texts, show_progress = show_progress)
        return embeddings

    async def __insert_points(self,
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
        collection_name = collection_name if collection_name != None else self._collection_name
        # Upload point
        await self._client.upsert(collection_name = collection_name,
                                  points = models.Batch(
                                      ids = point_ids,
                                      payloads = list_payloads,
                                      vectors = list_embeddings
                                  ))

    async def insert_documents(self,
                               documents :Sequence[BaseNode],
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

        # Get content and its embedding
        contents = [doc.get_content() for doc in documents]
        embeddings = None

        # Define dense embedding
        if isinstance(self._dense_embedding_model, BaseEmbedding):
            # With LlamaIndex Embedding
            # Get embedding model name
            model_name = self._dense_embedding_model.model_name
            # Define embedding
            embeddings = await self.__get_embeddings(texts = contents,
                                                     embedding_model = self._dense_embedding_model,
                                                     batch_size = embedded_batch_size,
                                                     num_workers = embedded_num_workers)

            # Get embedding dimension
            embedding_dimension = len(embeddings[0])
            # Define vector config
            dense_vectors_config = VectorParams(size = embedding_dimension,
                                                distance = self._distance,
                                                on_disk = self._on_disk,
                                                hnsw_config = models.HnswConfigDiff(on_disk = self._on_disk))
        else:
            # When embedding model is str, default is activated with FastEmbed model
            dense_vectors_config = self._client.get_fastembed_vector_params()
            # Get params
            model_name = list(dense_vectors_config)[0]

        sparse_embedding_model = None
        # Hybrid Search enabled
        if self._hybrid_search:
            sparse_embedding_model = self._client.get_fastembed_sparse_vector_params()

        # Define payloads
        payloads = self._convert_documents_to_payloads(documents = documents,
                                                       embedding_model_name = model_name)

        # Create collection if doesn't exist!
        await self.__create_collection(collection_name = self._collection_name,
                                       dense_vectors_config = dense_vectors_config,
                                       sparse_vectors_config = sparse_embedding_model,
                                       shard_number = self._shard_number,
                                       quantization_mode = self._quantization_mode,
                                       default_segment_number = self._default_segment_number)

        # Insert vector to collection
        if isinstance(self._dense_embedding_model, BaseEmbedding):
            # With BaseEmbedding model
            await self.__insert_points(list_embeddings = embeddings,
                                       list_payloads = payloads,
                                       batch_size = upload_batch_size,
                                       parallel = upload_parallel)
        else:
            # With FastEmbed Model
            await self._client.add(collection_name = self._collection_name,
                                   documents = contents,
                                   metadata = payloads,
                                   batch_size = upload_batch_size,
                                   parallel = upload_parallel)

    async def reembedding_with_collection(self,
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
        embeddings = await self.__get_embeddings(texts = contents,
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
        # Define vector config
        dense_vectors_config = VectorParams(size = embedding_dimension,
                                            distance = self._distance,
                                            on_disk = self._on_disk,
                                            hnsw_config = models.HnswConfigDiff(on_disk=self._on_disk))
        # Create collection if doesn't exist!
        await self.__create_collection(collection_name = collection_name,
                                       dense_vectors_config = dense_vectors_config)

        # Insert
        await self.__insert_points(collection_name = collection_name,
                                   list_embeddings = embeddings,
                                   list_payloads = payloads,
                                   point_ids = points_ids,
                                   batch_size = upload_batch_size)

    async def __search(self,
                       query_vector: List[Num],
                       filter : Optional[models.Filter] = None,
                       similarity_top_k :int = 3,
                       rescore :bool = True) -> List[types.ScoredPoint]:
        """
        Search and return top-k result from input embedding vector

        Args:
            query_vector (List[Num]): List of value represent for sematic embedding of query.
            filter (Filter): Filter the result under conditions.
            similarity_top_k (int): Determine the number of result should be returned.
            rescore (bool): Disable rescoring, which will reduce the number of disk reads, but slightly decrease the precision
        Returns:
            List[types.ScoredPoint]
        """

        # Check collection
        if not await self._client.collection_exists(self._collection_name):
            raise Exception(f"Collection {self._collection_name} isn't existed!")

        # Search params
        # search_params = models.SearchParams(hnsw_ef=512, exact=False)
        search_params = None
        # Disable rescore method
        if not rescore:
            search_params = models.SearchParams(
                quantization = models.QuantizationSearchParams(rescore = False)
            )

        # Return search
        result = await self._client.search(collection_name = self._collection_name,
                                           query_vector = query_vector,
                                           limit = similarity_top_k,
                                           search_params = search_params,
                                           query_filter = filter)
        return result

    async def retrieve(self,
                       query :str,
                       similarity_top_k :int = 3) -> Sequence[NodeWithScore]:
        """
        Retrieve nodes from vector store corresponding to question.

        :parameter query: The query str for retrieve.
        :type query: str
        :parameter similarity_top_k: Default is 3. Return top-k element from retrieval.
        :type similarity_top_k: int
        :return: Return a sequence of NodeWithScore
        :rtype Sequence[NodeWithScore]
        """

        # Check collection
        if not self._client.collection_exists(collection_name = self._collection_name):
            raise Exception(f"Collection {self._collection_name} not existed")

        # With LlamaIndex Embedding case
        if isinstance(self._dense_embedding_model, BaseEmbedding):
            # Get query embedding
            query_embedding = await self._dense_embedding_model.aget_query_embedding(query = query)
            # Get nodes
            scored_points = await self.__search(query_vector = query_embedding,
                                                similarity_top_k = similarity_top_k)
            # Convert to node with score
            return self._convert_score_point_to_node_with_score(scored_points=scored_points)
        else:
            # With FastEmbed model
            # Get nodes
            scored_points = await self._client.query(query_text = query,
                                                     collection_name = self._collection_name,
                                                     limit = similarity_top_k)
            # Convert to node with score
            return self._convert_query_response_to_node_with_score(scored_points = scored_points)

    async def _collection_info(self) -> types.CollectionInfo:
        """Return collection info"""
        return await self._client.get_collection(self._collection_name)

    async def _count_points(self) -> int:
        """Return the total amount of point inside collection"""
        # Get total amount of points
        result = await self._client.count(self._collection_name)
        return result.count

    async def _get_points(self,
                          limit :Optional[int] = "all",
                          with_vector :bool = False) -> Tuple[List[types.Record], Optional[types.PointId]]:
        """
        Get all the point in the Qdrant collection or with limited amount

        Args:
            limit (int, optional): The number of point retrieved. Default is all.
            with_vector (bool): Whether return vector or not.
        """
        # Get total point
        total_points = await self._count_points()

        # Limit if specify
        if limit == "all": limit = total_points
        # Return point
        result = await self._client.scroll(collection_name = self._collection_name,
                                           limit = limit,
                                           with_vectors = with_vector)
        return result