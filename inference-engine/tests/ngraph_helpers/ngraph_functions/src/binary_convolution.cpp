// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/data_utils.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makeBinaryConvolution(const Output<Node> &in,
                                            const std::vector<size_t> &filterSize,
                                            const std::vector<size_t> &strides,
                                            const std::vector<ptrdiff_t> &padsBegin,
                                            const std::vector<ptrdiff_t> &padsEnd,
                                            const std::vector<size_t> &dilations,
                                            const op::PadType &autoPad,
                                            size_t numOutChannels,
                                            float padValue,
                                            const std::vector<int8_t> &filterWeihgts) {
    auto shape = in.get_shape();
    std::vector<size_t> filterWeightsShape = {numOutChannels, shape[1]};
    filterWeightsShape.insert(filterWeightsShape.end(), filterSize.begin(), filterSize.end());
    auto filterWeightsNode = std::make_shared<op::Constant>(element::u1, filterWeightsShape);
    const size_t byteNum = (ngraph::shape_size(filterWeightsShape) + 7) / 8;
    int8_t *buffer = const_cast<int8_t *>(filterWeightsNode->get_data_ptr<int8_t>());
    if (filterWeihgts.size() == 0) {
        std::vector<int8_t> weihgts = NGraphFunctions::Utils::generateVector<element::Type_t::i8>(byteNum);
        for (size_t i = 0; i < byteNum; i++)
            buffer[i] = weihgts[i];
    } else {
        for (size_t i = 0; i < byteNum; i++)
            buffer[i] = filterWeihgts[i];
    }
    auto conv = std::make_shared<opset1::BinaryConvolution>(in, filterWeightsNode, strides, padsBegin, padsEnd, dilations,
                                                            opset1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT, padValue, autoPad);
    return conv;
}

}  // namespace builder
}  // namespace ngraph
