import logging
from dotenv import load_dotenv
from utils.k8s_config import get_secret_value

# Configure logging

logging.basicConfig(level=logging.DEBUG)  # Changed to DEBUG for more detailed logging
logger = logging.getLogger(__name__)

def load_environment():
    """
    Load environment variables from a .env file
    """
    if os.path.exists('.env'):
        load_dotenv(dotenv_path='.env', override=True)
        logger.debug("Loaded environment variables from '.env'")
    elif os.path.exists('.env.local'):
        load_dotenv(dotenv_path='.env.local', override=True)
        logger.debug("Loaded environment variables from '.env.local'")
    else:
        logger.warning("No .env or .env.local file found")

def get_env_variable(name: str, default: str = None, from_k8s_secret: bool = True, secret_name: str = None, secret_key: str = None) -> str:
    """
    Get the environment variables values with a priority to Kubernetes secrets then .env file
    """
    if from_k8s_secret and secret_name and secret_key:
        try:
            value = get_secret_value(secret_name, secret_key)
            logger.info(f"Environment variable '{name}' loaded from Kubernetes secret '{secret_name}' key '{secret_key}'")
            return value
        except KeyError:
            logger.error(f"Failed to load '{name}' from Kubernetes secret '{secret_name}' key '{secret_key}'")
    
    value = os.getenv(name, default)
    if value is not None:
        logger.info(f"Environment variable '{name}' loaded with value: {value}")
    else:
        logger.error(f"Required environment variable '{name}' not set.")
        raise EnvironmentError(f"Required environment variable '{name}' not set.")
    return value