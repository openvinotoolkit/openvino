// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "openvino/op/avg_pool.hpp"
#include "openvino/op/max_pool.hpp"
#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<Node> makePooling(const ov::Output<Node>& in,
                                  const std::vector<size_t>& strides,
                                  const std::vector<size_t>& padsBegin,
                                  const std::vector<size_t>& padsEnd,
                                  const std::vector<size_t>& kernel,
                                  const op::RoundingType& roundingType,
                                  const op::PadType& padType,
                                  bool excludePad,
                                  const ov::test::utils::PoolingTypes& poolType) {
    std::shared_ptr<ov::Node> pooling;
    switch (poolType) {
    case ov::test::utils::PoolingTypes::MAX:
        pooling = std::make_shared<ov::op::v1::MaxPool>(in, strides, padsBegin, padsEnd, kernel, roundingType, padType);

        break;
    case ov::test::utils::PoolingTypes::AVG:
        pooling = std::make_shared<ov::op::v1::AvgPool>(in,
                                                        strides,
                                                        padsBegin,
                                                        padsEnd,
                                                        kernel,
                                                        excludePad,
                                                        roundingType,
                                                        padType);
        break;
    }
    return pooling;
}
}  // namespace builder
}  // namespace ngraph
