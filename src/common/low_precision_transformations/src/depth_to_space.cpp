// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/depth_to_space.hpp"

#include <memory>
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "low_precision/network_helper.hpp"
#include "itt.hpp"

using namespace ov::pass::low_precision;

DepthToSpaceTransformation::DepthToSpaceTransformation(const Params& params) : TransparentBaseTransformation(params) {
    MATCHER_SCOPE(DepthToSpaceTransformation);
    auto matcher = pattern::wrap_type<ov::opset1::DepthToSpace>({ pattern::wrap_type<ov::opset1::Multiply>() });

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

bool DepthToSpaceTransformation::canBeTransformed(const std::shared_ptr<ov::Node>& layer) const {
    if (!LayerTransformation::canBeTransformed(layer)) {
        return false;
    }

    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(layer, defaultPrecisions);
    if (dequantization.multiply != nullptr) {
        if (!NetworkHelper::isScalarLike(dequantization.multiplyConstant)) {
            return false;
        }
    }

    if (dequantization.subtract != nullptr) {
        if (!NetworkHelper::isScalarLike(dequantization.subtractConstant)) {
            return false;
        }
    }

    return true;
}
