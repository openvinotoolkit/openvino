// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include <low_precision/lpt_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "low_precision/rt_info/shared_value_attribute.hpp"
#include "low_precision/layer_transformation.hpp"
#include "attribute_parameters.hpp"

namespace ngraph {
/**
 * @ingroup ie_transformation_common_api
 * @brief PerTensorQuantizationAttribute defines if operation input port requires per-tensor quantization.
 *
 * For more details about the attribute, refer to
 * [PerTensorQuantizationAttribute](@ref openvino_docs_OV_UG_lpt_PerTensorQuantization) page in the Inference Engine Developer Guide.
 */
class LP_TRANSFORMATIONS_API PerTensorQuantizationAttribute : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("LowPrecision::PerTensorQuantization", "", ov::RuntimeAttribute, 0);
    ~PerTensorQuantizationAttribute();
};
} // namespace ngraph
