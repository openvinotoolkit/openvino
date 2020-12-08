#include <inference_engine.hpp>
#include "ie_plugin_config.hpp"
#include "hetero/hetero_plugin_config.hpp"


int main() {
using namespace InferenceEngine;
//! [part3]
using namespace InferenceEngine::PluginConfigParams;
using namespace InferenceEngine::HeteroConfigParams;

// ...
InferenceEngine::Core core;
core.SetConfig({ { KEY_HETERO_DUMP_GRAPH_DOT, YES } }, "HETERO");
//! [part3]
return 0;
}
