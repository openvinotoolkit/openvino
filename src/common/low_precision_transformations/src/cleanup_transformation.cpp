// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/cleanup_transformation.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/disable_cleanup_attribute.hpp"

namespace ov {
namespace pass {
namespace low_precision {

CleanupTransformation::CleanupTransformation(const Params& params) : LayerTransformation(params) {
}

bool CleanupTransformation::canBeTransformed(const std::shared_ptr<Node>& layer) const {
    return canBeTransformedStatic(layer);
}

bool CleanupTransformation::canBeTransformedStatic(const std::shared_ptr<Node>& layer, const std::vector<ov::element::Type>& defaultPrecisions) {
    return getAttribute<DisableCleanupAttribute>(layer).empty();
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ov
