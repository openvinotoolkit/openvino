#include <openvino/runtime/core.hpp>

int main() {
//! [part1]
ov::Core core;
auto device_priorites = core.get_property("HETERO", ov::device::priorities);
//! [part1]
return 0;
}
