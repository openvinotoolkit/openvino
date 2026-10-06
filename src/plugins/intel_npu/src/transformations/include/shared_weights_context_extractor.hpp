// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/core/weight_sharing_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/aligned_buffer.hpp"

namespace ov {
namespace intel_npu {
namespace transformations {

struct SharedWeightsContextExtractor {
    using WeightSharingContextPtr = std::shared_ptr<ov::weight_sharing::Context>;
    using SharedConstant = std::shared_ptr<ov::op::v0::Constant>;
    using SharedSourceAndConstants = std::pair<std::shared_ptr<ov::AlignedBuffer>, std::vector<SharedConstant>>;
    using SharedSourcesWithConstants = std::vector<SharedSourceAndConstants>;

    static WeightSharingContextPtr extract_weight_sharing_context(const std::shared_ptr<ov::Model>& model);
    static WeightSharingContextPtr extract_weight_sharing_context(
        const SharedSourcesWithConstants& collected_sources_with_constants);
};

}  // namespace transformations
}  // namespace intel_npu
}  // namespace ov
