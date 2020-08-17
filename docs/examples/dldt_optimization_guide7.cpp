#include <inference_engine.hpp>
#include "ie_plugin_config.hpp"

#include "hetero/hetero_plugin_config.hpp"


int main() {
using namespace InferenceEngine;
	//these two networks go thru same plugin (aka device) and their requests will not overlap.

		auto executable_network0 = plugin.LoadNetwork(network0, {{PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS, PluginConfigParams::YES}});

		auto executable_network1 = plugin.LoadNetwork(network1, {{PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS, PluginConfigParams::YES}});

return 0;
}
