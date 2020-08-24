#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
//! [part0]
InferenceEngine::InferencePlugin plugin = InferenceEngine::PluginDispatcher({ FLAGS_pp }).getPluginByDevice(FLAGS_d);
//! [part0]
return 0;
}
