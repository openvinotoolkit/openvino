#include <openvino/runtime/core.hpp>

int main() {
ov::Core core;
auto model = core.read_model("sample.xml");

//! [compile_model]
{
    auto compiled_model = core.compile_model(model, "GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
}
//! [compile_model]

//! [compile_model_no_auto_batching]
{
    auto compiled_model = core.compile_model(model, "GPU", {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
                                                            ov::hint::allow_auto_batching(false)});
}
//! [compile_model_no_auto_batching]

return 0;
}
