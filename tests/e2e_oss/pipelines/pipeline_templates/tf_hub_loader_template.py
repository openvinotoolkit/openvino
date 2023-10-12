def tf_hub_loader(model_name=None,
                  model_link=None,
                  loader_timeout=300):
    return "load_model", {"load_tf_hub_model": {
        "model_name": model_name,
        'model_link': model_link,
        "loader_timeout": loader_timeout,
    }}
