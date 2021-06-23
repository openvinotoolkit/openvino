#include <ie_core.hpp>

int main() {
using namespace InferenceEngine;
    std::string modelPath = "/tmp/myModel.xml";
    std::string device = "GNA";
    std::map<std::string, std::string> deviceConfig;
//! [part2]
    InferenceEngine::Core ie;                                  // Step 1: create Inference engine object
    ie.SetConfig({{CONFIG_KEY(CACHE_DIR), "myCacheFolder"}});  // Step 1b: Enable caching
    ie.LoadNetwork(modelPath, device, deviceConfig);           // Step 2: LoadNetwork by model file path
//! [part2]
return 0;
}
