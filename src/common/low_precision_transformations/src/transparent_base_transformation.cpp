// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/transparent_base_transformation.hpp"

#include <memory>
#include <vector>

#include "openvino/util/log.hpp"
#include "low_precision/network_helper.hpp"

using namespace ov;
using namespace ov::pass;
using namespace ov::pass::low_precision;

bool TransparentBaseTransformation::transform(ov::pass::pattern::Matcher &m) {
    std::shared_ptr<Node> op = m.get_match_root();
    if (!canBeTransformed(op)) {
        return false;
    }

    op = NetworkHelper::separateInStandaloneBranch(op, defaultPrecisions);
    const auto newOperation = moveDequantizationAfter(op, NetworkHelper::getDequantization(op, defaultPrecisions));

    OPENVINO_DEBUG("LPT: done: ", newOperation);
    return true;
}

bool TransparentBaseTransformation::canBeTransformed(const std::shared_ptr<Node>& layer) const {
    if (!LayerTransformation::canBeTransformed(layer)) {
        return false;
    }

    const auto dequantization = NetworkHelper::getDequantization(layer, defaultPrecisions);
    if (dequantization.multiply == nullptr) {
        return false;
    }

    // If dequantization is placed on constant path and it doesn't change precision,
    // there is no point in the DQ propagation since it can be constant folded
    if (ov::is_type<ov::op::v0::Constant>(dequantization.data.get_node_shared_ptr()) &&
        dequantization.data.get_element_type() == dequantization.multiply->get_output_element_type(0)) {
        return false;
    }
    return true;
}

bool TransparentBaseTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}
