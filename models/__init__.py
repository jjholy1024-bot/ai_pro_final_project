from .my_model import MyModel


def get_model(name: str, **kwargs):
    models = {
        "MyModel": MyModel,
    }

    if name not in models:
        raise ValueError(f"Model '{name}' is not supported. Available: {list(models.keys())}")

    return models[name](**kwargs)
