// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/brgemm_blocking.hpp"
#include "transformations/tpp/x64/op/brgemm.hpp"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace pass {
/**
 * @interface BrgemmTPPBlocking
 * @brief Covers BrgemmTPP with blocking loops
 * @ingroup snippets
 */

class BrgemmTPPBlocking : public ov::snippets::lowered::pass::BrgemmBlocking<ov::intel_cpu::tpp::op::BrgemmTPP> {
public:
    OPENVINO_RTTI("BrgemmTPPBlocking", "BrgemmBlockingBase")

private:
    std::tuple<size_t, size_t, size_t> get_blocking_params(const ov::snippets::lowered::ExpressionPtr& brgemm_expr) override;
};

}  // namespace pass
}  // namespace tpp
}  // namespace intel_cpu
}  // namespace ov