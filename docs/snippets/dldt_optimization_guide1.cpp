#include <inference_engine.hpp>
#include "ie_plugin_config.hpp"
#include "hetero/hetero_plugin_config.hpp"


int main() {
using namespace InferenceEngine;
//! [part1]
Core ie;
auto netReader = ie.ReadNetwork("sample.xml");
InferenceEngine::InputsDataMap info(netReader.getInputsInfo());
auto& inputInfoFirst = info.begin()->second;
info.setInputPrecision(Precision::U8);
//! [part1]

return 0;
}
