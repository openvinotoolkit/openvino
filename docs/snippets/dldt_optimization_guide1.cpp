#include <ie_core.hpp>

int main() {
using namespace InferenceEngine;
//! [part1]
Core ie;
auto netReader = ie.ReadNetwork("sample.xml");
InferenceEngine::InputsDataMap info(netReader.getInputsInfo());
auto& inputInfoFirst = info.begin()->second;
for (auto& it : info) {
    it.second->setPrecision(Precision::U8);
}
//! [part1]

return 0;
}
