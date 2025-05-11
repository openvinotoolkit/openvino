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
    return true;
}

bool TransparentBaseTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}
