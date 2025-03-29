// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "simple_low_precision_transformer.hpp"

#include "low_precision/align_quantization_parameters.hpp"
#include "low_precision/layer_transformation.hpp"
#include "low_precision/low_precision.hpp"
#include "low_precision/markup_bias.hpp"
#include "low_precision/markup_can_be_quantized.hpp"
#include "low_precision/markup_quantization_granularity.hpp"

// cleanup transformations
#include "low_precision/convert.hpp"
#include "low_precision/eliminate_fake_quantize.hpp"
#include "low_precision/fold_convert.hpp"
#include "low_precision/fold_fake_quantize.hpp"
#include "low_precision/fuse_convert.hpp"
#include "low_precision/fuse_multiply_to_fake_quantize.hpp"
#include "low_precision/fuse_subtract_to_fake_quantize.hpp"
#include "low_precision/multiply_to_group_convolution.hpp"

#include <string>

using namespace testing;
using namespace ov::pass;
using namespace ov::pass::low_precision;

SimpleLowPrecisionTransformer::SimpleLowPrecisionTransformer(
    const std::vector<ov::pass::low_precision::PrecisionsRestriction>& precisionRestrictions,
    const std::vector<ov::pass::low_precision::QuantizationGranularityRestriction>& quantizationRestrictions,
    const AttributeParameters& params,
    const bool addCleanup) {
    auto passConfig = get_pass_config();

    // TODO: use one pass manager
    markup = std::make_shared<ov::pass::Manager>(passConfig);
    markup->register_pass<ov::pass::low_precision::MarkupCanBeQuantized>(params.defaultPrecisions);
    markup->register_pass<ov::pass::low_precision::MarkupPrecisions>(precisionRestrictions,
                                                                         params.defaultPrecisions);
    markup->register_pass<ov::pass::low_precision::MarkupQuantizationGranularity>(quantizationRestrictions);
    markup->register_pass<ov::pass::low_precision::MarkupAvgPoolPrecisionPreserved>(params.defaultPrecisions);
    markup->register_pass<ov::pass::low_precision::PropagatePrecisions>(params);
    markup->register_pass<ov::pass::low_precision::AlignQuantizationIntervals>(params.defaultPrecisions);
    markup->register_pass<ov::pass::low_precision::AlignQuantizationParameters>(params.defaultPrecisions);
    markup->register_pass<ov::pass::low_precision::MarkupBias>();

    common = std::make_shared<ov::pass::Manager>(passConfig);
    commonGraphRewrite = common->register_pass<ov::pass::GraphRewrite>();

    cleanup = common->register_pass<ov::pass::GraphRewrite>();
    if (addCleanup) {
        ov::pass::low_precision::LayerTransformation::Params params;
        cleanup->add_matcher<EliminateFakeQuantizeTransformation>(params);
        cleanup->add_matcher<FoldConvertTransformation>(params);
        cleanup->add_matcher<FuseConvertTransformation>(params);
        cleanup->add_matcher<FuseSubtractToFakeQuantizeTransformation>(params);
        cleanup->add_matcher<FuseMultiplyToFakeQuantizeTransformation>(params);

        cleanup->add_matcher<MultiplyToGroupConvolutionTransformation>(
            params,
            PrecisionsRestriction::getPrecisionsByOperationType<opset1::GroupConvolution>(precisionRestrictions));
    }
}

void SimpleLowPrecisionTransformer::transform(std::shared_ptr<ov::Model>& model) {
    run_on_model(model);
}

bool SimpleLowPrecisionTransformer::run_on_model(const std::shared_ptr<ov::Model>& model) {
    ov::pass::low_precision::TypeRelaxedReplacer pass;
    pass.run_on_model(model);

    markup->run_passes(model);
    common->run_passes(model);
    return true;
}
