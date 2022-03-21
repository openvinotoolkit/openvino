#include <openvino/openvino.hpp>

int main() {
{

//! [part3]
ov::Core core;

// Read a network in IR, PaddlePaddle, or ONNX format:
std::shared_ptr<ov::Model> model = core.read_model("sample.xml");

// compile a model on AUTO with Performance Hints enabled:
// To use the “throughput” mode:
ov::CompiledModel compiled_model = core.compile_model(model, "AUTO",
    ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
// or the “latency” mode:
ov::CompiledModel compiled_mode2 = core.compile_model(model, "AUTO",
    ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
//! [part3]
}
    return 0;
}
