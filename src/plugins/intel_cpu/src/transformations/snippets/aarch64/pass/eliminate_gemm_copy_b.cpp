// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eliminate_gemm_copy_b.hpp"

#include <memory>

#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/snippets/aarch64/op/gemm_copy_b.hpp"

namespace ov::intel_cpu::pass::aarch64 {

std::shared_ptr<ov::Node> EliminateGemmCopyB::get_copy_b_pattern(const std::shared_ptr<ov::Node>& input) const {
    return ov::pass::pattern::wrap_type<ov::intel_cpu::aarch64::GemmCopyB>({input});
}

bool EliminateGemmCopyB::is_supported_copy_b(const std::shared_ptr<ov::Node>& node) const {
    OPENVINO_ASSERT(ov::is_type<ov::intel_cpu::aarch64::GemmCopyB>(node), "EliminateGemmCopyB expects GemmCopyB node");
    return true;
}

}  // namespace ov::intel_cpu::pass::aarch64
