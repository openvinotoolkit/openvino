#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
//! [part0]
InferenceEngine::Core core;
std::vector<std::string> availableDevices = core.GetAvailableDevices();
//! [part0]
return 0;
}
