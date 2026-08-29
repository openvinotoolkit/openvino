// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eliminate_brgemm_copy_b.hpp"

#include <memory>

#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/snippets/x64/op/brgemm_copy_b.hpp"

namespace ov::intel_cpu::pass::x64 {

std::shared_ptr<ov::Node> EliminateBrgemmCopyB::get_copy_b_pattern(const std::shared_ptr<ov::Node>& input) const {
    return ov::pass::pattern::wrap_type<ov::intel_cpu::BrgemmCopyB>({input});
}

bool EliminateBrgemmCopyB::is_supported_copy_b(const std::shared_ptr<ov::Node>& node) const {
    const auto copy_b = ov::as_type_ptr<ov::intel_cpu::BrgemmCopyB>(node);
    OPENVINO_ASSERT(copy_b, "EliminateBrgemmCopyB expects BrgemmCopyB node");
    // TODO [157340]: support external repacking for copyB with compensations.
    return !copy_b->get_config().with_compensations();
}

}  // namespace ov::intel_cpu::pass::x64
