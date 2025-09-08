#include <openvino/runtime/core.hpp>

int main() {
    ov::Core core;
    auto model = core.read_model("sample.xml");
{

//! [compile_model]
auto compiled_model = core.compile_model(model, "GPU",
    ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
//! [compile_model]
}

{
//! [compile_model_no_auto_batching]
// disabling the automatic batching
// leaving intact other configurations options that the device selects for the 'throughput' hint 
auto compiled_model = core.compile_model(model, "GPU", 
    ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
    ov::hint::allow_auto_batching(false));
//! [compile_model_no_auto_batching]
}

{
//! [query_optimal_num_requests]
// when the batch size is automatically selected by the implementation
// it is important to query/create and run the sufficient #requests
auto compiled_model = core.compile_model(model, "GPU",
    ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
auto num_requests = compiled_model.get_property(ov::optimal_number_of_infer_requests);
//! [query_optimal_num_requests]
}

{
//! [hint_num_requests]
// limiting the available parallel slack for the 'throughput' hint via the ov::hint::num_requests
// so that certain parameters (like selected batch size) are automatically accommodated accordingly 
auto compiled_model = core.compile_model(model, "GPU",
    ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
    ov::hint::num_requests(4));
//! [hint_num_requests]
}

//! [hint_plus_low_level]
{
    // high-level performance hints are compatible with low-level device-specific settings 
auto compiled_model = core.compile_model(model, "CPU",
    ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
    ov::inference_num_threads(4));
}
//! [hint_plus_low_level]

    return 0;
}
