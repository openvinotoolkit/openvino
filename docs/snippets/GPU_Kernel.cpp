#include <ie_core.hpp>

int main() {
using namespace InferenceEngine;
//! [part0]
InferenceEngine::Core core;
// Load GPU Extensions
core.SetConfig({ { InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, "<path_to_the_xml_file>" } }, "GPU");
//! [part0]

return 0;
}
