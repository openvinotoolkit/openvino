// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations/cpu_opset/common/op/batch_gather_matmul_compressed.hpp"

namespace ov::intel_cpu {

class ConvertBatchGatherMatmulToBatchGatherMatmulCompressed : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertBatchGatherMatmulToBatchGatherMatmulCompressed");

    using SupportsPredicate =
        std::function<bool(const std::shared_ptr<ov::intel_cpu::BatchGatherMatmulCompressed>&, size_t, size_t, size_t)>;

    ConvertBatchGatherMatmulToBatchGatherMatmulCompressed(
        const std::vector<ov::element::Type>& supported_activation_types,
        const std::vector<ov::element::Type>& supported_weights_types,
        const SupportsPredicate& supports_config = nullptr,
        bool convert_u4zp_to_u8 = false);
};

}  // namespace ov::intel_cpu