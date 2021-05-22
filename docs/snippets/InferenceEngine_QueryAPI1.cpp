#include <inference_engine.hpp>
#include <ie_plugin_config.hpp>
#include "hetero/hetero_plugin_config.hpp"

int main() {
using namespace InferenceEngine;
//! [part1]
InferenceEngine::Core core;
bool dumpDotFile = core.GetConfig("HETERO", HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)).as<bool>();
//! [part1]
return 0;
}
