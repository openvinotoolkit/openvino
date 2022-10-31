// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include <legacy/details/ie_cnn_network_iterator.hpp>

namespace CommonTestUtils {

IE_SUPPRESS_DEPRECATED_START

std::shared_ptr<InferenceEngine::CNNLayer>
getLayerByName(const InferenceEngine::CNNNetwork & network, const std::string & layerName) {
    InferenceEngine::details::CNNNetworkIterator i(network), end;
    while (i != end) {
        auto layer = *i;
        if (layer->name == layerName)
            return layer;
        ++i;
    }
    IE_THROW(NotFound) << "Layer " << layerName << " not found in network";
}

IE_SUPPRESS_DEPRECATED_END

std::ostream& operator<<(std::ostream & os, OpType type) {
    switch (type) {
        case OpType::SCALAR:
            os << "SCALAR";
            break;
        case OpType::VECTOR:
            os << "VECTOR";
            break;
        default:
            IE_THROW() << "NOT_SUPPORTED_OP_TYPE";
    }
    return os;
}

std::vector<ov::OpSet> opsets;

void prepareOpsets(void) {
    if (opsets.size() == 0) {
        opsets.push_back(ov::get_opset1());
        opsets.push_back(ov::get_opset2());
        opsets.push_back(ov::get_opset3());
        opsets.push_back(ov::get_opset4());
        opsets.push_back(ov::get_opset5());
        opsets.push_back(ov::get_opset6());
        opsets.push_back(ov::get_opset7());
        opsets.push_back(ov::get_opset8());
        opsets.push_back(ov::get_opset9());
        opsets.push_back(ov::get_opset10());
    }
}

std::vector<ov::OpSet> getOpSets(void) {
    prepareOpsets();

    return opsets;
}

std::string getOpVersion(const ov::NodeTypeInfo& type_info) {
    prepareOpsets();

    for (size_t i = 0; i < opsets.size(); i++) {
        if (opsets[i].contains_type(type_info)) {
            return std::to_string(i + 1);
        }
    }
    return "undefined";
}

}  // namespace CommonTestUtils
