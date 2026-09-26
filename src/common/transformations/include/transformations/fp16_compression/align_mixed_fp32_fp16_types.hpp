// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type/element_type.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API AlignMixedFP32FP16Types;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief AlignMixedFP32FP16Types adds Converts to keep mixed FP16/FP32 graph type consistent
 * @param target lower precision the marked subgraphs must not be converted to (f16 by default).
 */
class ov::pass::AlignMixedFP32FP16Types : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("AlignMixedFP32FP16Types");
    explicit AlignMixedFP32FP16Types(const ov::element::Type& target = ov::element::f16);
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

private:
    ov::element::Type m_target;
};
