#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
core.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, "GPU");

return 0;
}
