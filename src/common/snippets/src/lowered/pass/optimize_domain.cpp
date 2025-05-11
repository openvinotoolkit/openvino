// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/optimize_domain.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/shape_inference/shape_inference.hpp"
#include "snippets/utils/utils.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

OptimizeDomain::OptimizeDomain(size_t& tile_rank) : Pass(), m_tile_rank(tile_rank) {
}
size_t OptimizeDomain::optimize(std::vector<VectorDims>& input_shapes,
                                  VectorDims& master_shape,
                                  const size_t total_work_amount,
                                  const size_t min_parallel_work_amount,
                                  const size_t min_jit_work_amount) {
    if (master_shape.size() <= 2)
        return false;

    auto CollapseLastDim = [](VectorDims& dims) {
        OPENVINO_ASSERT(dims.size() >= 2, "CollapseLastDim can't process shape with less than two dims");
        dims[dims.size() - 1] *= dims[dims.size() - 2];
        for (auto i = dims.size() - 2; i > 0; i--)
            dims[i] = dims[i - 1];
        dims[0] = 1;
    };
    // Check that neither of the two last dims is broadcasted, so they can be collapsed
    auto LastDimsNotBroadcasted = [] (const std::vector<VectorDims>& input_shapes, const VectorDims& master_shape) {
        const auto master_last = *master_shape.rbegin();
        const auto master_prelast = *++master_shape.rbegin();
        return std::all_of(input_shapes.begin(), input_shapes.end(),
                           [=](const VectorDims& s) {
                               OPENVINO_ASSERT(s.size() >= 2, "LastDimsNotBroadcasted can't process shape with less than two dims");
                               return *s.rbegin() == master_last &&
                                      *++s.rbegin() == master_prelast;
                            });
    };

    // Index of InputShape with the minimal rank
    size_t min_rank_idx = 0;
    for (size_t i = 1; i < input_shapes.size(); ++i) {
        if (input_shapes[i].size() < input_shapes[min_rank_idx].size())
            min_rank_idx = i;
    }

    size_t jit_work_amount = master_shape.back();
    size_t num_dims_collapsed = 0;
    while (jit_work_amount < min_jit_work_amount &&
           (num_dims_collapsed + 1) < input_shapes[min_rank_idx].size() &&
           can_increase_jit_work_amount(master_shape, min_parallel_work_amount, total_work_amount) &&
           LastDimsNotBroadcasted(input_shapes, master_shape)) {
        for (auto &s : input_shapes)
            CollapseLastDim(s);

        CollapseLastDim(master_shape);
        num_dims_collapsed++;

        jit_work_amount = master_shape.back();
    }
    return num_dims_collapsed;
}

inline bool OptimizeDomain::can_increase_jit_work_amount(const VectorDims& master_shape,
                                                         const size_t min_parallel_work_amount,
                                                         const size_t total_work_amount) {
    return master_shape.size() > 2 &&
           master_shape[master_shape.size() - 1] * master_shape[master_shape.size() - 2] *
           min_parallel_work_amount <= total_work_amount;
}
bool OptimizeDomain::run(snippets::lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::OptimizeDomain")
    const auto& config = linear_ir.get_config();
    if (linear_ir.empty())
        return false;

    m_tile_rank = 1;

    if (!config.m_enable_domain_optimization) {
        // Note: this is a special case: if optimization is not allowed, always assume 2D tile
        m_tile_rank = 2;
        return false;
    }

    VectorDims master_shape = linear_ir.get_master_shape();
    if (linear_ir.is_dynamic()) {
        // [134873] In dynamic case we don't know exact shapes so we cannot be sure to set tile_rank = 2:
        //          there can be really big shapes in inference stage which do not fit into the cache.
        m_tile_rank = 1;
        return false;
    }

    OPENVINO_ASSERT(config.m_min_parallel_work_amount != 0, "OptimizeDomain: Min parallel work amount can't equal to zero");
    std::vector<VectorDims> input_shapes;
    bool blocked_input_shapes = false;
    for (const auto& param : linear_ir.get_parameters()) {
        auto consumer_inputs = param->get_output_port_connector(0)->get_consumers();
        const auto& first_consumer = consumer_inputs.begin()->get_expr();
        if (auto rank_norm = as_type_ptr<op::RankNormalization>(first_consumer->get_node())) {
            // If RankNormalization appends dims, then the appended dims will be broadcasted
            // so collapsing is not allowed. We may increment tile rank though.
            if (rank_norm->get_num_append() != 0)
                blocked_input_shapes = true;
            // If RankNormalization prepends dims, then the dims should be ignored during domain optimization
            // to avoid passing already incremented shapes to linear_ir.shape_infer()
        }
        const ExpressionPtr& shape_producing_expr = blocked_input_shapes ?
                                                    first_consumer :
                                                    param;
        const auto& shape = utils::get_preordered_vdims(shape_producing_expr->get_output_port(0));
        OPENVINO_ASSERT(std::none_of(shape.begin(), shape.end(),
                                    [](size_t d) { return utils::is_dynamic_value(d); }),
                        "OptimizeDomain pass does not support dynamic shapes");
        input_shapes.emplace_back(shape);
    }
    const auto total_work_amount = std::accumulate(master_shape.begin(),
                                                   master_shape.end(),
                                                   (size_t)1,
                                                   std::multiplies<size_t>());
    const auto num_dims_collapsed = blocked_input_shapes ?
                                    0 :
                                    optimize(input_shapes,
                                              master_shape,
                                              total_work_amount,
                                              config.m_min_parallel_work_amount,
                                              config.m_min_kernel_work_amount);
    if (num_dims_collapsed > 0) {
        std::vector<VectorDimsRef> infer_shapes;
        infer_shapes.reserve(input_shapes.size());
        for (const auto& is : input_shapes)
            infer_shapes.emplace_back(is);
        // Need to propagate updated shapes through LIR
        linear_ir.shape_infer(infer_shapes);
    }
    // We can still try to increment tile rank after dimension collapsing
    if (can_increase_jit_work_amount(master_shape, config.m_min_parallel_work_amount, total_work_amount) &&
        num_dims_collapsed != master_shape.size() - 1)
        m_tile_rank++;
    return num_dims_collapsed > 0;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov