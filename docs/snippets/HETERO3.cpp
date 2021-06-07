#include <ie_core.hpp>

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
