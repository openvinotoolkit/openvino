#include <inference_engine.hpp>
#include <multi/multi_device_config.hpp>

int main() {
using namespace InferenceEngine;
        Core ie;
        std::string allDevices = "MULTI:";
        std::vector<std::string> availableDevices = ie.GetAvailableDevices();
        for (auto && device : availableDevices) {
            allDevices += device;
            allDevices += ((device == availableDevices[availableDevices.size()-1]) ? "" : ",");
        }
        ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, allDevices, {});
return 0;
}
