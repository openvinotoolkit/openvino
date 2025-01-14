// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/shape_types.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface Canonicalization
 * @brief Canonicalization inserts RankNormalization (ov::op::Unsqueeze analogue) operations to account for:
 *  - input ranks mismatch, then inputs with smaller ranks are prepeneded with 1
 *  - layouts mismatch (only planar + blocked is supported), planar shapes are postpended with 1
 *  @ingroup snippets
 */
class Canonicalization: public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("snippets::pass::Canonicalization");
    using BlockedShapeVector = op::Subgraph::BlockedShapeVector;
    using Layout = std::vector<size_t>;
    explicit Canonicalization(const BlockedShapeVector& blocked_input_shapes);
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    std::vector<VectorDims> m_in_shapes;
    std::vector<Layout> m_in_layouts;
    bool m_has_dynamic_inputs = false;
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
