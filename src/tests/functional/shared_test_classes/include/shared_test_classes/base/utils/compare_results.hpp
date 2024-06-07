// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


namespace ov {
namespace test {
namespace utils {

using CompareMap = std::map<ov::NodeTypeInfo, std::function<void(
        const std::shared_ptr<ov::Node> &node,
        size_t port,
        const ov::element::Type& inference_precision,
        const ov::Tensor &expected,
        const ov::Tensor &actual,
        double absThreshold,
        double relThreshold,
        double topk_threshold,
        double mvn_threshold)>>;

CompareMap getCompareMap();

} // namespace utils
} // namespace test
} // namespace ov
