#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
InferenceEngine::Core core;

std::vector<std::string> availableDevices = ie.GetAvailableDevices();

return 0;
}
