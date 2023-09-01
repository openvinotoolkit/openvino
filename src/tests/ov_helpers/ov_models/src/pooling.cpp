// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ov_models/builders.hpp"

namespace ov {
namespace builder {

std::shared_ptr<Node> makePooling(const ov::Output<Node> &in,
                                  const std::vector<size_t> &strides,
                                  const std::vector<size_t> &padsBegin,
                                  const std::vector<size_t> &padsEnd,
                                  const std::vector<size_t> &kernel,
                                  const op::RoundingType &roundingType,
                                  const op::PadType &padType,
                                  bool excludePad,
                                  const ov::helpers::PoolingTypes &poolType) {
    std::shared_ptr<ov::Node> pooling;
    switch (poolType) {
        case ov::helpers::PoolingTypes::MAX:
            pooling = std::make_shared<ov::opset3::MaxPool>(in, strides, padsBegin, padsEnd, kernel, roundingType,
                                                                padType);

            break;
        case ov::helpers::PoolingTypes::AVG:
            pooling = std::make_shared<ov::opset3::AvgPool>(in, strides, padsBegin, padsEnd, kernel,
                                                                excludePad,
                                                                roundingType, padType);
            break;
    }
    return pooling;
}

std::shared_ptr<Node> makeMaxPoolingV8(const ov::Output<Node> &in,
                                       const std::vector<size_t> &strides,
                                       const std::vector<size_t> &dilation,
                                       const std::vector<size_t> &padsBegin,
                                       const std::vector<size_t> &padsEnd,
                                       const std::vector<size_t> &kernel,
                                       const op::RoundingType &roundingType,
                                       const op::PadType &padType,
                                       const ov::element::Type &indexElementType,
                                       const int64_t axis) {
    std::shared_ptr<ov::Node> pooling = std::make_shared<ov::opset8::MaxPool>(in, strides, dilation, padsBegin, padsEnd,
                                                                                      kernel, roundingType, padType,
                                                                                      indexElementType, axis);
    return pooling;
}

}  // namespace builder
}  // namespace ov
