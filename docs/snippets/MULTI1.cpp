#include <ie_core.hpp>

int main() {
using namespace InferenceEngine;
//! [part1]
    Core ie; 
    auto network = ie.ReadNetwork("sample.xml");
    ExecutableNetwork exec = ie.LoadNetwork(network, "MULTI:HDDL,GPU", {});
    //...
    exec.SetConfig({{"MULTI_DEVICE_PRIORITIES", "GPU,HDDL"}});
    // you can even exclude some device
    exec.SetConfig({{"MULTI_DEVICE_PRIORITIES", "GPU"}});
    //...
    // and then return it back
    exec.SetConfig({{"MULTI_DEVICE_PRIORITIES", "GPU,HDDL"}});
    //but you cannot add new devices on the fly, the next line will trigger the following exception: 
    //[ ERROR ] [NOT_FOUND] You can only change device priorities but not add new devices with the Network's SetConfig(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES.
    //CPU device was not in the original device list!
    exec.SetConfig({{"MULTI_DEVICE_PRIORITIES", "CPU,GPU,HDDL"}});
//! [part1]
return 0;
}
