#ifndef INFERENCEENGINE_BRIDGE_INPUT_OUTPUT_INFO_H
#define INFERENCEENGINE_BRIDGE_INPUT_OUTPUT_INFO_H

#include "ie_input_info.hpp"

namespace InferenceEngineBridge {
    struct DataInfo {
        void setPrecision(std::string precision);

        InferenceEngine::InputInfo::Ptr actual;
        std::vector<std::size_t> dims;
        std::string precision;
        std::string layout;

    };

    struct InputInfo : public DataInfo {
        void setLayout(std::string layout);
    };

    struct OutputInfo : public DataInfo {
    };
}
#endif //INFERENCEENGINE_BRIDGE_INPUT_OUTPUT_INFO_H
