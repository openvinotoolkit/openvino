#include <ie_core.hpp>

int main() {
using namespace InferenceEngine;
    std::string modelPath = "/tmp/myModel.xml";
    std::string device = "GNA";
    std::map<std::string, std::string> deviceConfig;
//! [part1]
    InferenceEngine::Core ie;                                 // Step 1: create Inference engine object
    ie.LoadNetwork(modelPath, device, deviceConfig);          // Step 2: LoadNetwork by model file path
//! [part1]
return 0;
}
