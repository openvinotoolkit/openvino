#include <inference_engine.hpp>
#include <multi-device/multi_device_config.hpp>


int main() {
using namespace InferenceEngine;
//! [part3]
    Core ie;
    auto cnnNetwork = ie.ReadNetwork("sample.xml");
    std::string allDevices = "MULTI:";
    std::vector<std::string> myriadDevices = ie.GetMetric("MYRIAD", METRIC_KEY(myriadDevices));
    for (int i = 0; i < myriadDevices.size(); ++i) {
        allDevices += std::string("MYRIAD.")
                                + myriadDevices[i]
                                + std::string(i < (myriadDevices.size() -1) ? "," : "");
    }
    ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, allDevices, {});
//! [part3]
return 0;
}
