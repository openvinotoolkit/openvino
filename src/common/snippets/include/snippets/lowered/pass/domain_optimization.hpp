// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/pass/pass.hpp"
#include "snippets/shape_types.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface DomainOptimization
 * @brief Collapse input/output dimensions to balance parallel/per-thread load
 * @ingroup snippets
 */

class DomainOptimization : public snippets::lowered::pass::Pass {
    size_t& m_tile_rank;
public:
    OPENVINO_RTTI("DomainOptimization", "Pass")
    explicit DomainOptimization(size_t& tile_rank);
    bool run(LinearIR& linear_ir) override;
    static bool optimize(std::vector<VectorDims>& input_shapes,
                         VectorDims& master_shape,
                         size_t total_work_amount,
                         size_t min_parallel_work_amount,
                         size_t min_jit_work_amount);
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov