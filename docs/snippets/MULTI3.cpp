#include <openvino/openvino.hpp>

int main() {
//! [part3]
ov::Core core;
std::vector<std::string> myriadDevices = core.get_property("MYRIAD", ov::available_devices);
std::string all_devices;
for (size_t i = 0; i < myriadDevices.size(); ++i) {
    all_devices += std::string("MYRIAD.")
                            + myriadDevices[i]
                            + std::string(i < (myriadDevices.size() -1) ? "," : "");
}
ov::CompiledModel compileModel = core.compile_model("sample.xml", "MULTI",
    ov::device::priorities(all_devices));
//! [part3]
return 0;
}
