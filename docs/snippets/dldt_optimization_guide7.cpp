#include <ie_core.hpp>

int main() {
InferenceEngine::Core core;
auto network0 = core.ReadNetwork("sample.xml");
auto network1 = core.ReadNetwork("sample.xml");
//! [part7]
//these two networks go thru same plugin (aka device) and their requests will not overlap.
auto executable_network0 = core.LoadNetwork(network0, "CPU",
    {{InferenceEngine::PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS, InferenceEngine::PluginConfigParams::YES}});
auto executable_network1 = core.LoadNetwork(network1, "GPU",
    {{InferenceEngine::PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS, InferenceEngine::PluginConfigParams::YES}});
//! [part7]
return 0;
}
