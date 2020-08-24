#include <inference_engine.hpp>
#include "ie_plugin_config.hpp"
#include "hetero/hetero_plugin_config.hpp"


int main() {
using namespace InferenceEngine;
//! [part1]
InferenceEngine::InputsDataMap info(netReader.getNetwork().getInputsInfo());
auto& inputInfoFirst = info.begin()->second;
info->setInputPrecision(Precision::U8);
//! [part1]

return 0;
}
