#include <openvino/runtime/core.hpp>

int main() {
//! [part2]
ov::Core core;
core.set_property("CPU", ov::hint::inference_precision(ov::element::f32));
//! [part2]

return 0;
}
