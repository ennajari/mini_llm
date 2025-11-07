import yaml
class Config:
    def __init__(self, path="config.yaml"):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        self.__dict__.update(cfg)
