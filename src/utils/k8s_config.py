import os
import logging
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from urllib3.exceptions import MaxRetryError, NewConnectionError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_k8s_config():
    """
    Load Kubernetes configuration.
    """
    try:
        config.load_incluster_config()  # Load config from within a cluster
        logger.info("Loaded Kubernetes in-cluster configuration")
    except config.ConfigException:
        try:
            config.load_kube_config()  # Load config from kubeconfig file
            logger.info("Loaded Kubernetes local configuration")
        except config.ConfigException as e:
            logger.warning("Could not load Kubernetes configuration: not running in a cluster or kubeconfig file not found.")
            raise e

def get_secret_value(secret_name: str, key: str, namespace: str = 'default') -> str:
    """
    Get the value of a specific key from a Kubernetes secret.
    """
    v1 = client.CoreV1Api()
    try:
        secret = v1.read_namespaced_secret(secret_name, namespace)
        value = secret.data[key]
        logger.info(f"Loaded secret '{secret_name}' key '{key}' from namespace '{namespace}'")
        return value
    except (ApiException, MaxRetryError, NewConnectionError) as e:
        logger.error(f"Error retrieving secret '{secret_name}': {e}")
        raise KeyError(f"Secret '{secret_name}' or key '{key}' not found, or connection to Kubernetes API failed")

# Example usage in a controlled manner
if __name__ == "__main__":
    try:
        load_k8s_config()
        secret_value = get_secret_value("my-secret", "my-key")
        print(f"Secret Value: {secret_value}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
