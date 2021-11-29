#include <ie_core.hpp>

int main() {
//! [part3]
    InferenceEngine::Core ie;
    auto cnnNetwork = ie.ReadNetwork("sample.xml");
    std::string allDevices = "MULTI:";
    std::vector<std::string> myriadDevices = ie.GetMetric("MYRIAD", METRIC_KEY(AVAILABLE_DEVICES));
    for (size_t i = 0; i < myriadDevices.size(); ++i) {
        allDevices += std::string("MYRIAD.")
                                + myriadDevices[i]
                                + std::string(i < (myriadDevices.size() -1) ? "," : "");
    }
    InferenceEngine::ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, allDevices, {});
//! [part3]
return 0;
}
