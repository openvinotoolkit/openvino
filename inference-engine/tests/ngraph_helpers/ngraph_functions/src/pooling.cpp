// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makePooling(const ngraph::Output<Node> &in,
                                  const std::vector<size_t> &strides,
                                  const std::vector<size_t> &padsBegin,
                                  const std::vector<size_t> &padsEnd,
                                  const std::vector<size_t> &kernel,
                                  const op::RoundingType &roundingType,
                                  const op::PadType &padType,
                                  bool excludePad,
                                  const ngraph::helpers::PoolingTypes &poolType) {
    std::shared_ptr<ngraph::Node> pooling;
    switch (poolType) {
        case ngraph::helpers::PoolingTypes::MAX:
            pooling = std::make_shared<ngraph::opset3::MaxPool>(in, strides, padsBegin, padsEnd, kernel, roundingType,
                                                                padType);

            break;
        case ngraph::helpers::PoolingTypes::AVG:
            pooling = std::make_shared<ngraph::opset3::AvgPool>(in, strides, padsBegin, padsEnd, kernel,
                                                                excludePad,
                                                                roundingType, padType);
            break;
    }
    return pooling;
}

}  // namespace builder
}  // namespace ngraph
