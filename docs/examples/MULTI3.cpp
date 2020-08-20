#include <inference_engine.hpp>
#include <multi/multi_device_config.hpp>


int main() {
using namespace InferenceEngine;
//! [part3]
       Core ie;
        std::string allDevices = "MULTI:";
        std::vector<std::string> myriadDevices = ie->GetMetric("MYRIAD", METRIC_KEY(myriadDevices)));
        for (int i = 0; i < myriadDevices.size(); ++i) {
            allDevices += std::string("MYRIAD.")
                                  + myriadDevices[i]
                                  + std::string(i < (myriadDevices.size() -1) ? "," : "");
        }
        ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, allDevices, {});
//! [part3]
return 0;
}
