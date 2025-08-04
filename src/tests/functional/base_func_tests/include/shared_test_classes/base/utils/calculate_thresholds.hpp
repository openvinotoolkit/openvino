// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <vector>

#include "openvino/core/node.hpp"

#include "openvino/op/unsqueeze.hpp"

namespace ov {
namespace test {
namespace utils {

constexpr double DISABLE_THRESHOLD = -1.f;

struct Threshold {
    double abs_threshold, rel_threshold;

    Threshold(double in_abs_threshold = DISABLE_THRESHOLD, double in_rel_threshold = DISABLE_THRESHOLD)
    : abs_threshold(in_abs_threshold),
      rel_threshold(in_rel_threshold) {}
};

// the map define custom threshold per operation type
static std::map<ov::NodeTypeInfo, Threshold> custom_op_thresholds = {
        // NodeTypeInfo: {op_abs_threshold, op_rel_threshold}
        // if threshold is -1, the default approach will be used to calculate
        // Example
        // { ov::op::v0::FakeQuantize::get_type_info_static(), { 1e-2 }},
        // { ov::op::v0::Add::get_type_info_static(), { 1e-7, 1e-4 }},
};

std::pair<double, double> calculate_thresholds_by_model(
    const std::shared_ptr<ov::Model>& model,
    const std::shared_ptr<ov::Model>& ref_model = nullptr,
    const ov::element::Type& inference_precision = ov::element::dynamic);

} // namespace utils
} // namespace test
} // namespace ov
