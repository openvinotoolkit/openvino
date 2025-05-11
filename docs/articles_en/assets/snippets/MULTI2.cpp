#include <openvino/openvino.hpp>

int main() {
//! [part2]
ov::Core core;
std::shared_ptr<ov::Model> model = core.read_model("sample.xml");
std::vector<std::string> availableDevices = core.get_available_devices();
std::string all_devices;
for (auto && device : availableDevices) {
    all_devices += device;
    all_devices += ((device == availableDevices[availableDevices.size()-1]) ? "" : ",");
}
ov::CompiledModel compileModel = core.compile_model(model, "MULTI",
    ov::device::priorities(all_devices));
//! [part2]
return 0;
}
