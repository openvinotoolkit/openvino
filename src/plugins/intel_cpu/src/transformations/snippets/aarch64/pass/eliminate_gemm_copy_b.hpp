// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>

#include "emitters/snippets/input_repacker.hpp"
#include "openvino/core/node.hpp"
#include "transformations/snippets/common/pass/eliminate_copy_b.hpp"

namespace ov::intel_cpu::pass::aarch64 {

/**
 * @interface EliminateGemmCopyB
 * @brief AArch64 specialization of common CopyB elimination for GemmCopyB nodes.
 * @ingroup snippets
 */
class EliminateGemmCopyB : public ov::intel_cpu::pass::EliminateCopyB {
public:
    OPENVINO_MODEL_PASS_RTTI("EliminateGemmCopyB");
    explicit EliminateGemmCopyB(ov::intel_cpu::InputRepackerMap& input_repackers) : EliminateCopyB(input_repackers) {}

private:
    [[nodiscard]] std::shared_ptr<ov::Node> get_copy_b_pattern(const std::shared_ptr<ov::Node>& input) const override;
    [[nodiscard]] bool is_supported_copy_b(const std::shared_ptr<ov::Node>& node) const override;
};

}  // namespace ov::intel_cpu::pass::aarch64
