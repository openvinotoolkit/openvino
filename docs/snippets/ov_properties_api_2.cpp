#include <openvino/runtime/core.hpp>

int main() {
//! [part2]
ov::Core core;
auto cpu_device_name = core.get_property("GPU", ov::device::full_name);
//! [part2]
return 0;
}
