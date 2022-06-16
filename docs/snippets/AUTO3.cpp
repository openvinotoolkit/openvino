#include <openvino/openvino.hpp>

int main() {
{

//! [part3]
ov::Core core;

// Read a network in IR, PaddlePaddle, or ONNX format:
std::shared_ptr<ov::Model> model = core.read_model("sample.xml");

// Compile a model on AUTO with Performance Hint enabled:
// To use the “THROUGHPUT” option:
ov::CompiledModel compiled_model = core.compile_model(model, "AUTO",
    ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
// To use the “LATENCY” option:
ov::CompiledModel compiled_mode2 = core.compile_model(model, "AUTO",
    ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
// To use the “CUMULATIVE_THROUGHPUT” option:
ov::CompiledModel compiled_mode3 = core.compile_model(model, "AUTO",
    ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT));    
//! [part3]
}
    return 0;
}
