// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/reduce_min.hpp"
#include <memory>
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "low_precision/network_helper.hpp"
#include "itt.hpp"

namespace ov {
namespace pass {
namespace low_precision {

ReduceMinTransformation::ReduceMinTransformation(const Params& params) : ReduceBaseTransformation(params) {
    MATCHER_SCOPE(ReduceMinTransformation);
    auto matcher = pattern::wrap_type<ov::opset1::ReduceMin>({ pattern::wrap_type<ov::opset1::Multiply>(), pattern::wrap_type<ov::opset1::Constant>() });

    ov::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(m);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool ReduceMinTransformation::canBeTransformed(const std::shared_ptr<Node>& reduce) const {
    if (!ov::is_type<ov::opset1::ReduceMin>(reduce)) {
        return false;
    }

    if (!ReduceBaseTransformation::canBeTransformed(reduce)) {
        return false;
    }

    const auto dequantization = NetworkHelper::getDequantization(reduce, defaultPrecisions);
    const std::vector<float> scales = ov::as_type_ptr<ov::opset1::Constant>(dequantization.multiplyConstant)->cast_vector<float>();
    if (std::any_of(scales.begin(), scales.end(), [](const float value) { return value < 0.0; })) {
        return false;
    }

    return true;
}

bool ReduceMinTransformation::isPrecisionPreserved(std::shared_ptr<Node> reduce) const noexcept {
    return true;
}

bool ReduceMinTransformation::getUpdatePrecision(const std::shared_ptr<Node>& reduce) const {
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ov
