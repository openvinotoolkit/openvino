#include <ie_core.hpp>
#include "cldnn/cldnn_config.hpp"

int main() {
using namespace InferenceEngine;
//! [part0]
InferenceEngine::Core core;
// Load GPU Extensions
core.SetConfig({ { InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, "<path_to_the_xml_file>" } }, "GPU");
//! [part0]

//! [part1]
IE_SUPPRESS_DEPRECATED_START
core.SetConfig({ { PluginConfigParams::KEY_DUMP_KERNELS, PluginConfigParams::YES } }, "GPU");
IE_SUPPRESS_DEPRECATED_END
//! [part1]

return 0;
}
