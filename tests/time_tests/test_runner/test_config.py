test_cases = [
    {
        "device":
            {"name": "CPU"},
        "model":
            {"path": "/home/vurusovs_local/work/openvino/tests/stress_tests/tmp_model/vgg16.xml"}   # TODO: add link to `test_data` repo model
        },
    {
        "device":
            {"name": "GPU"},
        "model":
            {"name": "vgg16",
             "precision": "FP32",
             "source": "omz"}
    }
]
