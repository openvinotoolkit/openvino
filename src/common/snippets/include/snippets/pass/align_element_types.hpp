// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"
#include "snippets/op/subgraph.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface AlignElementTypes
 * @brief Align body precision with expected input/output precision. Insert op::ConvertSaturation if necessary.
 * @ingroup snippets
 */
class AlignElementTypes: public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("snippets::pass::AlignElementTypes");
    AlignElementTypes(std::vector<ov::element::Type> input_precisions,
                      std::vector<ov::element::Type> output_precisions);
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    std::vector<ov::element::Type> m_input_precisions;
    std::vector<ov::element::Type> m_output_precisions;
};

}  // namespace pass
}  // namespace snippets
}  // namespace ov
