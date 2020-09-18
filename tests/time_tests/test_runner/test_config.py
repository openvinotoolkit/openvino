test_cases = [
    {
        "device":
            {"name": "CPU"},
        "model":
            {"path": "/home/vurusovs_local/work/openvino/tests/stress_tests/tmp_model/vgg16.xml"},   # TODO: add link to `test_data` repo model
        "references":
            {'create_exenetwork': {'avg': 999505.0 * 1.2, 'stdev': 4587.0 * 1.2},
             'first_inference': {'avg': 73434.0 * 1.2, 'stdev': 88.0 * 1.2},
             'first_inference_latency': {'avg': 1005008.0 * 1.2, 'stdev': 4615.0 * 1.2},
             'full_run': {'avg': 1114453.0 * 1.2, 'stdev': 4231.0 * 1.2},
             'load_network': {'avg': 803487.0 * 1.2, 'stdev': 4513.0 * 1.2},
             'load_plugin': {'avg': 5479.0 * 1.2, 'stdev': 32.0 * 1.2},
             'read_network': {'avg': 195628.0 * 1.2, 'stdev': 663.0 * 1.2}}
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
