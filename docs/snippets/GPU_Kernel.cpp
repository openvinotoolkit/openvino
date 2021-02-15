#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
//! [part0]
InferenceEngine::Core core;
// Load GPU Extensions
core.SetConfig({ { InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, "<path_to_the_xml_file>" } }, "GPU");
//! [part0]

//! [part1]
core.SetConfig({ { PluginConfigParams::KEY_DUMP_KERNELS, PluginConfigParams::YES } }, "GPU");
//! [part1]

return 0;
}
