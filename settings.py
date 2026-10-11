import yaml


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_config("config.yaml")
temperature = config["temperature"]
seed = config["seed"]
top_p = config["top_p"]
