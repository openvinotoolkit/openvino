#include <openvino/openvino.hpp>

int main() {
using namespace InferenceEngine;
//! [part2]
ov::Core core;
std::shared_ptr<ov::Model> model = core.read_model("sample.xml");
std::string allDevices = "MULTI:";
std::vector<std::string> availableDevices = core.get_available_devices();
for (auto && device : availableDevices) {
    allDevices += device;
    allDevices += ((device == availableDevices[availableDevices.size()-1]) ? "" : ",");
}
ov::CompiledModel compileModel = core.compile_model(model, allDevices);
//! [part2]
return 0;
}
