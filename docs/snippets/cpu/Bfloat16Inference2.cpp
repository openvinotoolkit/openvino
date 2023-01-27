#include <openvino/runtime/core.hpp>

int main() {
using namespace InferenceEngine;
//! [part2]
ov::Core core;
core.set_property("CPU", ov::inference_precision(ov::element::f32));
//! [part2]

return 0;
}
