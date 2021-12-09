#include <ie_core.hpp>

int main() {
using namespace InferenceEngine;
//! [part0]
InferenceEngine::Core core;
std::vector<std::string> availableDevices = core.GetAvailableDevices();
//! [part0]
return 0;
}
