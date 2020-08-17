#include <inference_engine.hpp>

int main() {
using namespace InferenceEngine;
InferenceEngine::Core core;

bool dumpDotFile = core.GetConfig("HETERO", HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)).as<bool>();

return 0;
}
