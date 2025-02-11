// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/canonicalization.hpp"
#include "snippets/op/rank_normalization.hpp"
#include "snippets/itt.hpp"
#include "snippets/utils/utils.hpp"
#include "snippets/lowered/port_descriptor.hpp"

namespace ov {
namespace snippets {

pass::Canonicalization::Canonicalization(const BlockedShapeVector& blocked_input_shapes) {
    m_in_shapes.reserve(blocked_input_shapes.size());
    m_in_layouts.reserve(blocked_input_shapes.size());
    for (const auto& bs : blocked_input_shapes) {
        m_has_dynamic_inputs |= utils::is_dynamic_vdims(bs.first);
        m_in_shapes.emplace_back(bs.first);
        m_in_layouts.emplace_back(bs.second);
        // Note: Blocking (if any) must be accounted for in input shapes
        OPENVINO_ASSERT(m_in_shapes.back().size() == m_in_layouts.back().size(), "Input shapes and layouts must have the same rank");
    }
}

bool pass::Canonicalization::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_MODEL_SCOPE(Canonicalization);
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::Canonicalization")
    bool is_modified = false;
    const ParameterVector& params = m->get_parameters();
    OPENVINO_ASSERT(m_in_shapes.size() == params.size(),
                    "Number of parameters for snippet doesn't match passed to the Canonicalization pass. ",
                    "Expected: ", m_in_shapes.size(), " Got: ", params.size(), ".");

    // Note that shape rank also incorporates layout, so NCHW16c would have shape rank 5
    auto is_blocked_layout = [](const Layout& l) {
        return l.size() != std::set<size_t>(l.begin(), l.end()).size();
    };
    auto compare_ranks = [](const Layout& l, const Layout& r) {
        return l.size() < r.size();
    };
    // Layout with the max rank
    const auto& max_rank_it = std::max_element(m_in_layouts.begin(), m_in_layouts.end(), compare_ranks);
    Layout base_layout = *max_rank_it;
    size_t max_rank = base_layout.size();
    const bool base_is_blocked = is_blocked_layout(base_layout);

    for (size_t i = 0; i < m_in_layouts.size(); i++) {
        const auto& i_layout = m_in_layouts[i];
        const auto& i_shape = m_in_shapes[i];
        const auto i_rank = i_layout.size();
        const bool i_is_blocked = is_blocked_layout(i_layout);
        // Canonicalization logic briefly:
        // * If this input is blocked => Reshape corresponding input parameter, so the following transformations
        //   will work with a shape of a larger rank. In dynamic case, this shape will be updated during shapeInfer()
        //   call, but the important thing is that the shape rank won't change.
        // * If some of the input shapes is blocked (=> base_is_blocked), but this input is planar,
        //   then insert RankNormalization op after this input. This is needed, so all shapes inside the body have
        //   similar ranks.
        if (i_is_blocked) {
            OPENVINO_ASSERT(base_is_blocked && i_rank == max_rank, "If this shape is blocked, base must also be blocked");
            params[i]->set_partial_shape(snippets::utils::vdims_to_pshape(i_shape));
            is_modified = true;
        } else if (i_rank < max_rank) {
            size_t num_append = base_is_blocked;
            OPENVINO_ASSERT(max_rank >= i_rank + num_append, "Unsupported blocked shapes combination in canonicalization");
            size_t num_prepend = max_rank - i_rank - num_append;
            const auto& out = params[i]->output(0);
            const auto& target_inputs = out.get_target_inputs();
            auto rank_norm = std::make_shared<op::RankNormalization>(out, num_prepend, num_append);
            for (auto& in : target_inputs)
                in.replace_source_output(rank_norm);
            is_modified = true;
        } else {
            // todo: 4d blocked + 5d planar layouts are not supported: <N, C, H, W, c> + <N, C, D, H, W>
            OPENVINO_ASSERT(equal(base_layout.begin(), base_layout.end(), i_layout.begin()),
                            "Canonicalization got input shapes of equal ranks but different layouts, which is not supported");
        }
    }
    return is_modified;
}

} // namespace snippets
} // namespace ov