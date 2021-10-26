#include <ie_core.hpp>

int main() {
using namespace InferenceEngine;
//! [part2]
    Core ie;
    auto cnnNetwork = ie.ReadNetwork("sample.xml");
    std::string allDevices = "MULTI:";
    std::vector<std::string> availableDevices = ie.GetAvailableDevices();
    for (auto && device : availableDevices) {
        allDevices += device;
        allDevices += ((device == availableDevices[availableDevices.size()-1]) ? "" : ",");
    }
    ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, allDevices, {});
//! [part2]
return 0;
}
