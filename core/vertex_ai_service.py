import json
import os
import uuid
import google.auth
import vertexai
import logging
from google.cloud import aiplatform
from google.api_core import exceptions
from config.loader import AppConfig
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.vector_stores.vertexaivectorsearch import VertexAIVectorStore
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.llms.vertex import Vertex
from llama_index.core.storage.docstore import SimpleDocumentStore
from core.gcs_service import GCSService
from core import constants


logger = logging.getLogger(__name__)


class VertexAIService:
    def __init__(
        self,
        config: AppConfig,
        embed_model: VertexTextEmbedding = None,
        storage_service: GCSService = None,
    ):
        self.config = config
        self.credentials, self.project_id = google.auth.default()
        self.vector_store: VertexAIVectorStore = None
        self.embed_model = embed_model
        self.query_engine: BaseException = None
        self.gcs_service = storage_service

        self.cache_dir = constants.CACHE_DIR
        self.docstore_path = constants.CACHE_DOCSTORE_PATH

        self.docstore = SimpleDocumentStore()
        self.storage_context: StorageContext = None
        self.index: VectorStoreIndex = None

        logger.info("VertexAIService initialized (but not connected)")

    def connect_and_load(self):
        """Connects to services and loads the docstore into a StorageContext."""

        if self.storage_context is not None:
            logger.debug("StorageContext already initialized. Skipping connection.")
            return

        logger.info("Connecting to services and loading local state...")

        # If the docstore file exists, load it on startup.
        if os.path.exists(self.docstore_path):
            self.docstore = SimpleDocumentStore.from_persist_path(self.docstore_path)
            logger.info(f"Loaded {len(self.docstore.docs)} nodes from local docstore.")

        logger.info("   Connecting to Vertex AI services...")

        # Now we initialize the clients
        self.vector_store = self._get_vector_store()

        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            docstore=self.docstore,
        )

        logger.info(
            f"StorageContext created with {len(self.docstore.docs)} nodes in docstore."
        )

    # private helper methods for initialization
    def _get_vector_store(self):
        # initialize and return VertexAIVectorStore
        if not self.vector_store:
            self.vector_store = VertexAIVectorStore(
                project_id=self.config.gcp_project_id,
                region=self.config.gcp_region,
                index_id=self._vs_index.resource_name,
                endpoint_id=self.vs_endpoint.resource_name,
                gcs_bucket_name=self.config.gcs_bucket_name,
            )
        else:
            logger.info("Using existing VertexAIVectorStore instance.")

        # return the vector store instance
        return self.vector_store

    # Resource Provisioning - call from ingest script
    def provision_vertex_resources(self):
        """A high-level method to ensure all necessary GCP resources exist."""
        logger.info("Provisioning GCP resources...")
        self.gcs_service.ensure_bucket_exists()

        self._ensure_index_exists()
        self._ensure_endpoint_exists_and_index_is_deployed()
        logger.info("Resource provisioning complete.")

    def create_endpoint(
        self, endpoint_name: str
    ) -> aiplatform.MatchingEngineIndexEndpoint:
        """
        Creates a Vector Search endpoint.
        """
        endpoint_names = [
            endpoint.resource_name
            for endpoint in aiplatform.MatchingEngineIndexEndpoint.list(
                filter=f"display_name={endpoint_name}"
            )
        ]

        if len(endpoint_names) == 0:
            logger.info(f"Creating Vector Search index endpoint {endpoint_name} ...")
            vs_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
                display_name=endpoint_name, public_endpoint_enabled=True
            )
            logger.info(
                f"Vector Search index endpoint {vs_endpoint.display_name} created with resource name {vs_endpoint.resource_name}"
            )
        else:
            vs_endpoint = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name=endpoint_names[0]
            )
            logger.info(
                f"Vector Search index endpoint {vs_endpoint.display_name} exists with resource name {vs_endpoint.resource_name}"
            )

        return vs_endpoint

    def deploy_index_to_endpoint(
        self,
        vs_index: aiplatform.MatchingEngineIndex,
        vs_endpoint: aiplatform.MatchingEngineIndexEndpoint,
        index_name: str,
    ) -> aiplatform.MatchingEngineIndexEndpoint:
        """
        Deploys a Vector Search endpoint.
        """
        # check if endpoint exists
        index_endpoints = [
            (deployed_index.index_endpoint, deployed_index.deployed_index_id)
            for deployed_index in vs_index.deployed_indexes
        ]

        if len(index_endpoints) == 0:
            logger.info(
                f"Deploying Vector Search index {vs_index.display_name} at endpoint {vs_endpoint.display_name} ..."
            )
            vs_deployed_index = vs_endpoint.deploy_index(
                index=vs_index,
                deployed_index_id=index_name,
                display_name=index_name,
                machine_type="e2-standard-16",
                min_replica_count=1,
                max_replica_count=1,
            )
            logger.info(
                f"Vector Search index {vs_index.display_name} is deployed at endpoint {vs_deployed_index.display_name}"
            )
        else:
            vs_deployed_index = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name=index_endpoints[0][0]
            )
            logger.info(
                f"Vector Search index {vs_index.display_name} is already deployed at endpoint {vs_deployed_index.display_name}"
            )

        return vs_deployed_index

    def _find_index(self, display_name) -> aiplatform.MatchingEngineIndex:
        """
        Checks if a Vertex AI Vector Search index with the configured display name exists.
        Returns the index if found otherwise None.
        """
        found_index: aiplatform.MatchingEngineIndex = None
        indexes = [
            index
            for index in aiplatform.MatchingEngineIndex.list(
                filter=f"display_name={display_name}"
            )
        ]
        if len(indexes) > 0:
            found_index = indexes[0]

        return found_index

    def _ensure_index_exists(self):
        """
        Checks if a Vertex AI Vector Search index with the configured display name exists.
        If it does not, it creates a new one. Sets self.vs_index with the found or created index object.
        """
        display_name = self.config.vs_index_name
        logger.info(
            f"INFO:     Checking for existing Vector Search index named '{display_name}'..."
        )
        found_index = self._find_index(display_name)

        # desired_display_name = self.config.vs_index_name
        # parent_path = f"projects/{self.config.gcp_project_id}/locations/{self.config.gcp_region}"

        # logger.info(f" Checking for existing Vector Search index named '{desired_display_name}'...")

        # found_index = None
        # try:
        #     # List all indexes in the project/region
        #     existing_indexes = aiplatform.MatchingEngineIndex.list(location=self.config.gcp_region)

        #     # Loop through them to find one with a matching display name
        #     for index in existing_indexes:
        #         if index.display_name == desired_display_name:
        #             logger.info(f" Found existing index: {index.resource_name}")
        #             found_index = index
        #             break # Stop searching once we find a match

        # except exceptions.PermissionDenied as e:
        #     logger.info(f"ERROR:    Permission denied when trying to list indexes. Check IAM roles.")
        #     raise e

        # --- If the index was NOT found after checking, create it ---
        if not found_index:
            logger.info(
                f" No existing index found. Creating new index '{self.config.vs_index_name}'..."
            )

            # For a Tree-AH index, it needs an empty GCS directory to store its data.
            # We can use the bucket we already provisioned.
            index_contents_uri = f"gs://{self.config.gcs_bucket_name}/index-metadata/"

            try:
                new_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
                    project=self.config.gcp_project_id,
                    location=self.config.gcp_region,
                    display_name=self.config.vs_index_name,
                    contents_delta_uri=index_contents_uri,
                    dimensions=self.config.vs_dimensions,
                    distance_measure_type="DOT_PRODUCT_DISTANCE",
                    feature_norm_type="UNIT_L2_NORM",  # or "L2" for L2 normalization
                    shard_size="SHARD_SIZE_SMALL",
                    index_update_method="BATCH_UPDATE",  # allowed values BATCH_UPDATE , STREAM_UPDATE,
                    approximate_neighbors_count=150,  # A standard starting value
                    leaf_node_embedding_count=1000,  # A standard starting value
                    leaf_nodes_to_search_percent=10,  # A standard starting value
                    # sync=True, # Make this call wait until creation is complete. OBSOLETE?
                )
                logger.info(f"Waiting for index creation: {new_index.resource_name}")
                new_index.wait()  # Wait for the index to be fully created (NO NEED TO DO THIS)
                logger.info(
                    f"New index creation has completed: {new_index.resource_name}"
                )
                self._vs_index = new_index
            except exceptions.GoogleAPICallError as e:
                logger.info(f"ERROR:    Failed to create the index. {e}")
                raise e

        else:
            # If we found an existing index, set it to our instance variable.
            self._vs_index = found_index
            # vs_index = aiplatform.MatchingEngineIndex(index_name=index_names[0])

        return self._vs_index

    def _ensure_endpoint_exists_and_index_is_deployed(self):
        logger.info(
            f"Ensuring index is deployed to endpoint '{self.config.vs_index_endpoint_name}'..."
        )
        self.vs_endpoint = self.create_endpoint(
            endpoint_name=self.config.vs_index_endpoint_name
        )
        self.vs_deployed_index = self.deploy_index_to_endpoint(
            vs_index=self._vs_index,
            vs_endpoint=self.vs_endpoint,
            index_name=self.config.vs_index_name,
        )

    def ingest_nodes(self, nodes: list[Document]):
        """
        Performs a robust batch update to the Vertex AI index by taking
        direct control of the GCS staging and API call. This is necessary
        to ensure custom node IDs are respected and the correct GCS path is used.
        """
        if not self.storage_context:
            raise RuntimeError("Must connect before ingesting.")

        # The 'from_documents' method is also used for ingestion.
        # It's smart enough to see the nodes and add them to the stores
        # defined in the provided storage_context.
        logger.info(f"Ingesting {len(nodes)} nodes using the storage context...")

        # 1. Embed all the nodes
        node_texts = [node.get_content() for node in nodes]
        embeddings = self.embed_model.get_text_embedding_batch(node_texts)

        # 2. Prepare the JSONL data file as required by Vertex AI
        jsonl_data = []
        for i, node in enumerate(nodes):
            # --- THE FINAL FIX ---
            # We will manually create the 'restricts' structure that the
            # LlamaIndex retriever is looking for.

            # Serialize the entire node object to a JSON string
            node_json_string = node.to_json()

            # Create the restrict namespace entry
            restricts_payload = [
                {
                    "namespace": "_node_content", 
                    "allow": [node_json_string]
                }
            ]

            json_line_object = {
                "id": node.id_,
                "embedding": list(embeddings[i]),
                "restricts": restricts_payload,
            }

            jsonl_data.append(json.dumps(json_line_object))

        jsonl_content = "\n".join(jsonl_data)

        # 3. Upload the data to a correct staging location in GCS.
        #    We no longer need to guess the path. We will simply create a
        #    temporary directory for our batch.
        batch_id = uuid.uuid4()
        gcs_blob_name = f"ingestion-staging/{batch_id}/embeddings.json"

        # Use your GCS Service to upload the string
        self.gcs_service.upload_string(
            jsonl_content, destination_blob_name=gcs_blob_name
        )

        # The API requires the URI to the DIRECTORY containing the file(s).
        gcs_directory_uri = (
            f"gs://{self.config.gcs_bucket_name}/ingestion-staging/{batch_id}"
        )
        logger.info(f" Staged embedding data at: {gcs_directory_uri}")

        # 4. Call the low-level aiplatform.MatchingEngineIndex.update_embeddings method
        #    This is the modern name for the batch update operation.
        try:
            # Get the client for the specific index resource
            index_client = aiplatform.MatchingEngineIndex(
                index_name=self._vs_index.resource_name
            )

            # This is the correct, robust call.
            index_client.update_embeddings(
                contents_delta_uri=gcs_directory_uri,
            )
            logger.info("   Vertex AI index update job submitted successfully.")

        except Exception as e:
            logger.info(
                f"ERROR:    Direct API call to update_embeddings failed. Error: {e}"
            )
            raise

        self.docstore.add_documents(nodes)
        self.docstore.persist(persist_path=self.docstore_path)
        logger.info(
            f"Saved {len(nodes)} nodes to local docstore at {self.docstore_path}"
        )

        self.index = None
        logger.info(
            "Ingestion complete. Invalidated old index object. A new one will be created on next query."
        )

    def clear_index_data(self):
        """
        Removes all data points from the connected Vector Search index.
        This is a powerful operation and should be used with care.
        """
        if not self._vs_index:
            raise RuntimeError(
                "Index must be loaded before it can be cleared. Call provision_resources() first."
            )

        logger.info(
            f" Clearing all data points from index '{self._vs_index.display_name}'..."
        )

        # The 'remove_datapoints' method takes a list of data point IDs.
        # To clear the whole index, we would need to list all IDs first,
        # which can be complex.

        # A simpler and more brutal approach for a demo project is to delete
        # and recreate the index. This guarantees a perfectly clean slate.

        try:
            index_id = self._vs_index.resource_name
            endpoint_id = self.vs_endpoint.resource_name  # Assuming you have this

            # 1. Undeploy the index from the endpoint
            logger.info(f" Undeploying index from endpoint '{endpoint_id}'...")
            self.vs_endpoint.undeploy_index(deployed_index_id=self.config.vs_index_name)

            # 2. Delete the index
            logger.info(f" Deleting index '{index_id}'...")
            aiplatform.MatchingEngineIndex(index_name=index_id).delete()

            logger.info(
                "   Index deleted successfully. It will be recreated on the next provision call."
            )
            # Reset the local state
            self._vs_index = None
            self.is_connected = False

        except Exception as e:
            logger.info(
                f"ERROR:    Failed to clear index. It may need to be done manually in the GCP Console. Error: {e}"
            )
            # This can be complex if the index is still in use or has replicas.

    def delete_endpoint(self):
        endpoint_id = self.vs_endpoint.resource_name

        # Undeploy the index from the endpoint
        logger.info(f" Undeploying index from endpoint '{endpoint_id}'...")
        self.vs_endpoint.undeploy_index(deployed_index_id=self.config.vs_index_name)

    def delete_index(self):
        index_id = self._vs_index.resource_name
        self._vs_index.delete()
        # aiplatform.MatchingEngineIndex.delete(index_id)

    def get_index(self) -> VectorStoreIndex:
        """
        Lazily initializes and returns the high-level VectorStoreIndex object.
        This is the primary way the application should get the index for querying.
        """
        # If the index object hasn't been created yet for this session, create it.
        if self.index is None:
            if self.storage_context is None:
                raise RuntimeError(
                    "Must call connect_and_load() before the index can be created."
                )

            logger.info(
                "Initializing VectorStoreIndex from storage context for the first time..."
            )

            # This is the code you correctly identified. It builds the index
            # object from the already-prepared storage context.
            self.index = VectorStoreIndex.from_documents(
                [],
                storage_context=self.storage_context,
                embed_model=self.embed_model,
            )
            logger.info("Connection and loading complete.")

        return self.index

    def get_query_engine(self, llm: Vertex):
        """Builds and returns a LlamaIndex query engine."""
        if self.storage_context is None:
            raise RuntimeError(
                "Must call connect_and_load() before the index can be created."
            )
        if not self.vector_store:
            raise ValueError("Vector store is not initialized.")

        # Use the LlamaIndex Settings context manager to configure the LLMs
        Settings.llm = llm
        Settings.embed_model = self.embed_model

        logger.info(f"Building query engine for the index with {llm.model} LLM...")
        return self.index.as_query_engine(similarity_top_k=self.config.top_k_retrieval)

    # def query_index(self, query):
    #     # Logic to query the vector search index
    #     pass

    def _get_index_by_name(self):
        """
        Helper method to find an existing index by its display name.
        Returns the index object if found, otherwise None.
        """

        index_list = aiplatform.MatchingEngineIndex.list(
            filter=f"display_name={self.config.vs_index_name}"
        )

        index_names = [
            index.resource_name
            for index in aiplatform.MatchingEngineIndex.list(
                filter=f"display_name={self.config.vs_index_name}"
            )
        ]

        if len(index_names) > 0:
            return aiplatform.MatchingEngineIndex(index_name=index_names[0])

        return None

    def _get_endpoint_by_name(self):
        """
        Helper method to find an existing endpoint by its display name.
        Returns the endpoint object if found, otherwise None.
        """
        endpoint_names = [
            endpoint.resource_name
            for endpoint in aiplatform.MatchingEngineIndexEndpoint.list(
                filter=f"display_name={self.config.vs_index_endpoint_name}"
            )
        ]

        if len(endpoint_names) > 0:
            return aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name=endpoint_names[0]
            )

        return None

    def reset_resources(self):
        """
        Handles the complete and safe teardown of the Vertex AI Index and Endpoint.
        This is a destructive operation.
        """
        logger.info("--- Initiating resource reset ---")

        # First, we need to find the existing resources
        endpoint = self._get_endpoint_by_name()
        index = self._get_index_by_name()

        if not endpoint and not index:
            logger.info("No existing index or endpoint found. Nothing to reset.")
            return

        # --- Step 1: Undeploy the index from the endpoint (CRITICAL STEP) ---
        if endpoint and index:
            try:
                # It appears that index.name is the numerical ID we need.
                index_to_undeploy_id = index.name

                logger.info(
                    f"Checking for index ID '{index_to_undeploy_id}' on endpoint '{endpoint.display_name}'..."
                )

                # Flag to track if we found and undeployed the index
                was_undeployed = False

                # Loop through the indexes deployed at this endpoint
                for deployed_index in endpoint.deployed_indexes:
                    # deployed_index.index is the full resource name string.
                    # We need to extract its last part to get the numerical ID.
                    deployed_index_numerical_id = deployed_index.index.split("/")[-1]

                    if deployed_index_numerical_id == index_to_undeploy_id:
                        logger.info(f"Match found. Undeploying index from endpoint...")

                        # The undeploy call correctly uses the deployed_index.id,
                        # which your debugger has shown is the numerical ID.
                        endpoint.undeploy_index(deployed_index_id=deployed_index.id)

                        # The operation is asynchronous, so we should add a wait
                        # for robustness.
                        logger.info("Waiting for undeploy operation to complete...")
                        endpoint.wait()

                        logger.info("Undeploy successful.")
                        was_undeployed = True
                        break  # Exit the loop, our job is done here.

            # try:
            #     # Check if the index is actually deployed to this endpoint
            #     deployed_indexes = endpoint.deployed_indexes
            #     if any(di.id == index.name.split('/')[-1] for di in deployed_indexes):
            #         logger.info(f"Undeploying index '{index.display_name}' from endpoint '{endpoint.display_name}'...")
            #         endpoint.undeploy(deployed_index_id=deployed_indexes[0].id) # Assuming one index
            #         logger.info("Undeploy successful.")
            #     else:
            #         logger.info("Index is not deployed to this endpoint. Skipping undeploy.")
            except Exception as e:
                logger.info(
                    f"Warning: Could not undeploy index from endpoint. It may have already been undeployed. Error: {e}"
                )

        # --- Step 2: Delete the endpoint ---
        if endpoint:
            try:
                logger.info(f"Deleting endpoint '{endpoint.display_name}'...")
                endpoint.delete()
                logger.info("Endpoint deleted successfully.")
            except Exception as e:
                logger.info(f"Error deleting endpoint: {e}")

        # --- Step 3: Delete the index ---
        if index:
            try:
                logger.info(f"Deleting index '{index.display_name}'...")
                index.delete()
                logger.info("Index deleted successfully.")
            except Exception as e:
                logger.info(f"Error deleting index: {e}")

        # --- Step 4: Clear the internal state of the service object ---
        self._vs_index = None
        self._vs_endpoint = None
        logger.info("--- Resource reset complete ---")


if __name__ == "__main__":

    # Example usage
    config = AppConfig()  # Load your configuration

    vertexai.init(project=config.gcp_project_id, location=config.gcp_region)
    vertex = VertexAIService(config)

    logger.info(f"Initialized VertexAIService.  ProjectID = {vertex.project_id}")

    vertex.reset_resources()  # This will clear the index and endpoint if they exist
