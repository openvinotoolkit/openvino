#include <openvino/runtime/core.hpp>

int cpu_Bfloat16Inference0() {
//! [part0]
ov::Core core;
auto cpuOptimizationCapabilities = core.get_property("CPU", ov::device::capabilities);
//! [part0]
return 0;
}
