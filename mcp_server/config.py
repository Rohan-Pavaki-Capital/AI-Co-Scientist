import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# load environment variables from .env file
# First try project root .env, then mcp_server/.env
root_env_path = Path(__file__).parent.parent / '.env'
local_env_path = Path(__file__).parent / '.env'

if root_env_path.exists():
    load_dotenv(dotenv_path=root_env_path)
    logger.info(f"Loaded environment from {root_env_path}")

if local_env_path.exists():
    load_dotenv(dotenv_path=local_env_path, override=True)
    logger.info(f"Loaded environment from {local_env_path}")

if not root_env_path.exists() and not local_env_path.exists():
    logger.warning(f".env file not found at {root_env_path} or {local_env_path} - using system environment only")

# logging config
LOG_LEVEL = os.environ.get('COSCIENTIST_MCP_LOG_LEVEL') or os.environ.get('LOG_LEVEL', 'INFO')
LOG_LEVEL = LOG_LEVEL.upper()
