// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <tuple>
#include <vector>

namespace ov {
class AlignedBuffer;
class Any;
class Model;
namespace weight_sharing {
struct Context;
}
namespace intel_npu {
namespace transformations {

using SharedWeightsResult =
    std::tuple<std::vector<std::shared_ptr<ov::AlignedBuffer>>, std::shared_ptr<ov::weight_sharing::Context>>;

SharedWeightsResult assign_shared_weight_to_model_if_possible(const std::shared_ptr<ov::Model>& model,
                                                              const ov::Any& shared_weight_property);

}  // namespace transformations
}  // namespace intel_npu
}  // namespace ov
