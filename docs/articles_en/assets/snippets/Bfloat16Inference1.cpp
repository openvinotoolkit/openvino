#include <openvino/runtime/core.hpp>

int main() {
//! [part1]
ov::Core core;
auto network = core.read_model("sample.xml");
auto exec_network = core.compile_model(network, "CPU");
auto inference_precision = exec_network.get_property(ov::hint::inference_precision);
//! [part1]

return 0;
}
