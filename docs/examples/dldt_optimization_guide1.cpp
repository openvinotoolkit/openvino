#include <inference_engine.hpp>
#include "ie_plugin_config.hpp"
#include "hetero/hetero_plugin_config.hpp"

int main() {
using namespace InferenceEngine;
InferenceEngine::InputsDataMap info(netReader.getNetwork().getInputsInfo());
auto& inputInfoFirst = info.begin()->second;
info->setInputPrecision(Precision::U8);
return 0;
}
