#include <openvino/runtime/core.hpp>

int cpu_Bfloat16Inference2() {
using namespace InferenceEngine;
//! [part2]
ov::Core core;
core.set_property("CPU", ov::hint::inference_precision(ov::element::f32));
//! [part2]

return 0;
}
