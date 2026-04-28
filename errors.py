class ModelLoadError(Exception):
    def __init__(self, artifact_name: str, path: str, cause: Exception):
        super().__init__(f"Failed to load '{artifact_name}' from '{path}': {cause}")
