#include <openvino/openvino.hpp>

int main() {
//! [part3]
ov::Core core;
std::vector<std::string> GPUDevices = core.get_property("GPU", ov::available_devices);
std::string all_devices;
for (size_t i = 0; i < GPUDevices.size(); ++i) {
    all_devices += std::string("GPU.")
                            + GPUDevices[i]
                            + std::string(i < (GPUDevices.size() -1) ? "," : "");
}
ov::CompiledModel compileModel = core.compile_model("sample.xml", "MULTI",
    ov::device::priorities(all_devices));
//! [part3]
return 0;
}
