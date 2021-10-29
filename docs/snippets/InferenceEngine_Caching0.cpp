#include <ie_core.hpp>

int main() {
using namespace InferenceEngine;
    std::string modelPath = "/tmp/myModel.xml";
    std::string device = "GNA";
    std::map<std::string, std::string> deviceConfig;
//! [part0]
    InferenceEngine::Core ie;                                 // Step 1: create Inference engine object
    ie.SetConfig({{CONFIG_KEY(CACHE_DIR), "myCacheFolder"}}); // Step 1b: Enable caching
    auto cnnNet = ie.ReadNetwork(modelPath);                  // Step 2: ReadNetwork
    //...                                                     // Step 3: Prepare inputs/outputs
    //...                                                     // Step 4: Set device configuration
    ie.LoadNetwork(cnnNet, device, deviceConfig);             // Step 5: LoadNetwork
//! [part0]
return 0;
}
