#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
InferenceEngine::Core core;
core.SetConfig({ { CONFIG_KEY(ENFORCE_BF16), CONFIG_VALUE(NO) } }, "CPU");
return 0;
}
