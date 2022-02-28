#include <openvino/openvino.hpp>

int main() {
//! [part3]
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model("sample.xml");
    std::string allDevices = "MULTI:";
    std::vector<std::string> myriadDevices = core.get_property("MYRIAD", ov::available_devices);
    for (size_t i = 0; i < myriadDevices.size(); ++i) {
        allDevices += std::string("MYRIAD.")
                                + myriadDevices[i]
                                + std::string(i < (myriadDevices.size() -1) ? "," : "");
    }

    ov::CompiledModel compileModel = core.compile_model(model, allDevices, {});
//! [part3]
return 0;
}
