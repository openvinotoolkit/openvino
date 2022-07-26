// Copyright (C) 2018-2022 Intel Corporation
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
#include <low_precision/markup_quantization_granularity.hpp>
#include <low_precision/markup_can_be_quantized.hpp>

using namespace testing;
using namespace ngraph::pass;

OPENVINO_SUPPRESS_DEPRECATED_START

SimpleLowPrecisionTransformer::SimpleLowPrecisionTransformer(
    const std::vector<ngraph::pass::low_precision::PrecisionsRestriction>& precisionRestrictions,
    const std::vector<ngraph::pass::low_precision::QuantizationGranularityRestriction>& quantizationRestrictions,
    const AttributeParameters& params) {
    auto passConfig = get_pass_config();

    // TODO: use one pass manager
    markup = std::make_shared<ngraph::pass::Manager>(passConfig);
    markup->register_pass<ngraph::pass::low_precision::MarkupCanBeQuantized>(params.defaultPrecisions);
    markup->register_pass<ngraph::pass::low_precision::MarkupPrecisions>(precisionRestrictions, params.defaultPrecisions);
    markup->register_pass<ngraph::pass::low_precision::MarkupQuantizationGranularity>(quantizationRestrictions);
    markup->register_pass<ngraph::pass::low_precision::MarkupAvgPoolPrecisionPreserved>(params.defaultPrecisions);
    markup->register_pass<ngraph::pass::low_precision::PropagatePrecisions>(params);
    markup->register_pass<ngraph::pass::low_precision::AlignQuantizationIntervals>(params.defaultPrecisions);
    markup->register_pass<ngraph::pass::low_precision::AlignQuantizationParameters>(params.defaultPrecisions);

    common = std::make_shared<ngraph::pass::Manager>(passConfig);
    commonGraphRewrite = common->register_pass<ngraph::pass::GraphRewrite>();
    cleanup = common->register_pass<ngraph::pass::GraphRewrite>();
}

void SimpleLowPrecisionTransformer::transform(std::shared_ptr<ngraph::Function>& function) {
    run_on_model(function);
}

bool SimpleLowPrecisionTransformer::run_on_model(const std::shared_ptr<ngraph::Function>& function) {
    ngraph::pass::low_precision::TypeRelaxedReplacer pass;
    pass.run_on_model(function);

    markup->run_passes(function);
    common->run_passes(function);
    return true;
}
