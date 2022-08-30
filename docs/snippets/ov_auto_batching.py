from openvino.runtime import Core

core = Core()
model = core.read_model(model="sample.xml")

# [compile_model]
config = {"PERFORMANCE_HINT": "THROUGHPUT"}
compiled_model = core.compile_model(model, "GPU", config)
# [compile_model]

# [compile_model_no_auto_batching]
# disabling the automatic batching
# leaving intact other configurations options that the device selects for the 'throughput' hint 
config = {"PERFORMANCE_HINT": "THROUGHPUT",
          "ALLOW_AUTO_BATCHING": False}
compiled_model = core.compile_model(model, "GPU", config)
# [compile_model_no_auto_batching]

# [query_optimal_num_requests]
# when the batch size is automatically selected by the implementation
# it is important to query/create and run the sufficient requests
config = {"PERFORMANCE_HINT": "THROUGHPUT"}
compiled_model = core.compile_model(model, "GPU", config)
num_requests = compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
# [query_optimal_num_requests]

# [hint_num_requests]
config = {"PERFORMANCE_HINT": "THROUGHPUT",
          "PERFORMANCE_HINT_NUM_REQUESTS": "4"}
# limiting the available parallel slack for the 'throughput'
# so that certain parameters (like selected batch size) are automatically accommodated accordingly 
compiled_model = core.compile_model(model, "GPU", config)
# [hint_num_requests]

# [hint_plus_low_level]
config = {"PERFORMANCE_HINT": "THROUGHPUT",
          "INFERENCE_NUM_THREADS": "4"}
# limiting the available parallel slack for the 'throughput'
# so that certain parameters (like selected batch size) are automatically accommodated accordingly
compiled_model = core.compile_model(model, "CPU", config)
# [hint_plus_low_level]]