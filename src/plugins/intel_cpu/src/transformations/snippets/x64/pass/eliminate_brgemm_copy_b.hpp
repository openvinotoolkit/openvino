// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <set>
#include <utility>

#include "emitters/snippets/input_repacker.hpp"
#include "openvino/core/node.hpp"
#include "transformations/snippets/common/pass/eliminate_copy_b.hpp"

namespace ov::intel_cpu::pass::x64 {

/**
 * @interface EliminateBrgemmCopyB
 * @brief x64 specialization of common CopyB elimination for BrgemmCopyB nodes.
 * @ingroup snippets
 */
class EliminateBrgemmCopyB : public ov::intel_cpu::pass::EliminateCopyB {
public:
    OPENVINO_MODEL_PASS_RTTI("EliminateBrgemmCopyB");
    explicit EliminateBrgemmCopyB(ov::intel_cpu::InputRepackerMap& input_repackers,
                                  bool runtime_repacking_supported = true,
                                  std::set<size_t> compile_time_repacking_idxs = {})
        : EliminateCopyB(input_repackers, runtime_repacking_supported, std::move(compile_time_repacking_idxs)) {}

private:
    [[nodiscard]] std::shared_ptr<ov::Node> get_copy_b_pattern(const std::shared_ptr<ov::Node>& input) const override;
    [[nodiscard]] bool is_supported_copy_b(const std::shared_ptr<ov::Node>& node) const override;
};

}  // namespace ov::intel_cpu::pass::x64
