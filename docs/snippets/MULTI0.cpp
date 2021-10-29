#include <ie_core.hpp>

int main() {
using namespace InferenceEngine;
//! [part0]
    Core ie; 
    auto network = ie.ReadNetwork("sample.xml");
    //NEW IE-CENTRIC API, the "MULTI" plugin is (globally) pre-configured with the explicit option:
    ie.SetConfig({{"MULTI_DEVICE_PRIORITIES", "HDDL,GPU"}}, "MULTI");
    ExecutableNetwork exec0 = ie.LoadNetwork(network, "MULTI", {});

    //NEW IE-CENTRIC API, configuration of the "MULTI" is part of the network configuration (and hence specific to the network):
    ExecutableNetwork exec1 = ie.LoadNetwork(network, "MULTI", {{"MULTI_DEVICE_PRIORITIES", "HDDL,GPU"}});
    //NEW IE-CENTRIC API, same as previous, but configuration of the "MULTI" is part of the name (so config is empty), also network-specific:
    ExecutableNetwork exec2 = ie.LoadNetwork(network, "MULTI:HDDL,GPU", {});
//! [part0]
return 0;
}
