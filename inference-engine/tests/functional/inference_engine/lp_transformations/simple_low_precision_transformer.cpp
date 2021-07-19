// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "simple_low_precision_transformer.hpp"

#include <string>
#include <ngraph/ngraph.hpp>
#include <low_precision/low_precision.hpp>
#include <low_precision/transformation_context.hpp>
#include <low_precision/layer_transformation.hpp>
#include <low_precision/transformation_context.hpp>
#include <low_precision/low_precision.hpp>
#include <low_precision/align_quantization_parameters.hpp>
#include <low_precision/markup_per_tensor_quantization.hpp>
#include <low_precision/markup_can_be_quantized.hpp>

using namespace testing;
using namespace ngraph::pass;

SimpleLowPrecisionTransformer::SimpleLowPrecisionTransformer(
    const std::vector<ngraph::pass::low_precision::OperationPrecisionRestriction>& precisionRestrictions,
    const std::vector<ngraph::pass::low_precision::OperationPerTensorQuantizationRestriction>& quantizationRestrictions) {

    // TODO: use one pass manager
    markup = std::make_shared<ngraph::pass::Manager>();
    markup->register_pass<ngraph::pass::low_precision::MarkupCanBeQuantized>();
    markup->register_pass<ngraph::pass::low_precision::MarkupPrecisions>(precisionRestrictions);
    markup->register_pass<ngraph::pass::low_precision::MarkupPerTensorQuantization>(quantizationRestrictions);
    markup->register_pass<ngraph::pass::low_precision::MarkupAvgPoolPrecisionPreserved>();
    markup->register_pass<ngraph::pass::low_precision::PropagatePrecisions>();
    markup->register_pass<ngraph::pass::low_precision::AlignQuantizationIntervals>();
    markup->register_pass<ngraph::pass::low_precision::AlignQuantizationParameters>();

    common = std::make_shared<ngraph::pass::Manager>();
    commonGraphRewrite = common->register_pass<ngraph::pass::GraphRewrite>();
    cleanup = common->register_pass<ngraph::pass::GraphRewrite>();
}

void SimpleLowPrecisionTransformer::transform(std::shared_ptr<ngraph::Function>& function) {
    ngraph::pass::low_precision::TypeRelaxedReplacer pass;
    pass.run_on_function(function);

    markup->run_passes(function);
    common->run_passes(function);
}
