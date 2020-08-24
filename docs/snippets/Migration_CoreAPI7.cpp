#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
//! [part7]
core.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, "GPU");
//! [part7]
return 0;
}
