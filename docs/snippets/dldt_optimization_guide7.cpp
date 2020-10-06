#include <inference_engine.hpp>
#include "ie_plugin_config.hpp"
#include "hetero/hetero_plugin_config.hpp"


int main() {
using namespace InferenceEngine;
Core plugin;
auto network0 = plugin.ReadNetwork("sample.xml");
auto network1 = plugin.ReadNetwork("sample.xml");
//! [part7]
//these two networks go thru same plugin (aka device) and their requests will not overlap.
auto executable_network0 = plugin.LoadNetwork(network0, "CPU", {{PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS, PluginConfigParams::YES}});
auto executable_network1 = plugin.LoadNetwork(network1, "GPU", {{PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS, PluginConfigParams::YES}});
//! [part7]
return 0;
}
