#include <openvino/runtime/core.hpp>

int main() {
//! [part0]
ov::Core core;
auto cpuOptimizationCapabilities = core.get_property("CPU", ov::device::capabilities);
//! [part0]
return 0;
}
