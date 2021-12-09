#include <ie_core.hpp>

int main() {
using namespace InferenceEngine;
//! [part2]
InferenceEngine::Core core;
core.SetConfig({ { CONFIG_KEY(ENFORCE_BF16), CONFIG_VALUE(NO) } }, "CPU");
//! [part2]

return 0;
}
