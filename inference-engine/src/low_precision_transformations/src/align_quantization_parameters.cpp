// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/align_quantization_parameters.hpp"
#include <memory>
#include "low_precision/create_attribute.hpp"
#include "low_precision/propagate_through_precision_preserved.hpp"
#include "low_precision/rt_info/quantization_alignment_attribute.hpp"
#include "low_precision/rt_info/per_tensor_quantization_attribute.hpp"
#include "low_precision/update_shared_precision_preserved.hpp"

using namespace ov;
using namespace ov::pass::low_precision;

NGRAPH_RTTI_DEFINITION(ov::pass::low_precision::AlignQuantizationParameters, "AlignQuantizationParameters", 0);

bool ov::pass::low_precision::AlignQuantizationParameters::run_on_function(std::shared_ptr<ov::Function> f) {
    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    std::shared_ptr<ov::pass::GraphRewrite> propagation = manager.register_pass<ov::pass::GraphRewrite>();
    propagation->add_matcher<low_precision::CreateAttribute<QuantizationAlignmentAttributePtr>>();
    propagation->add_matcher<low_precision::PropagateThroughPrecisionPreserved<QuantizationAlignmentAttribute>>();
    propagation->add_matcher<low_precision::UpdateSharedPrecisionPreserved<QuantizationAlignmentAttributePtr, PerTensorQuantizationAttribute>>();
    manager.run_passes(f);
    return false;
}
