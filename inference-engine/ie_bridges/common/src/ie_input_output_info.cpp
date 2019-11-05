#include "ie_input_output_info.h"

#include "helpers.h"

void InferenceEngineBridge::DataInfo::setPrecision(std::string precision) {
    this->actual->setPrecision(InferenceEngineBridge::precision_map[precision]);
}

void InferenceEngineBridge::InputInfo::setLayout(std::string layout) {
    this->actual->setLayout(InferenceEngineBridge::layout_map[layout]);
}
