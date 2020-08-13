#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
InferenceEngine::InferencePlugin plugin = InferenceEngine::PluginDispatcher({ FLAGS_pp }).getPluginByDevice(FLAGS_d);
return 0;
}
