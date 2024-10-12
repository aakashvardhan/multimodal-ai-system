import yaml

def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Loaded configuration.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)