// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::snippets::pass {

/**
 * @interface AlignElementTypes
 * @brief Align body precision with expected input/output precision. Insert op::ConvertSaturation if necessary.
 * @ingroup snippets
 */
class AlignElementTypes : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("snippets::pass::AlignElementTypes");
    AlignElementTypes(std::vector<ov::element::Type> input_precisions,
                      std::vector<ov::element::Type> output_precisions);
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    std::vector<ov::element::Type> m_input_precisions;
    std::vector<ov::element::Type> m_output_precisions;
};

}  // namespace ov::snippets::pass
