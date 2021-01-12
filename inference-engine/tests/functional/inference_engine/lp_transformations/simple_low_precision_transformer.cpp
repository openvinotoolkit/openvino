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

    markup = std::make_shared<ngraph::pass::Manager>();
    markup->register_pass<ngraph::pass::low_precision::MarkupCanBeQuantized>();
    markup->register_pass<ngraph::pass::low_precision::MarkupPrecisions>(precisionRestrictions);
    markup->register_pass<ngraph::pass::low_precision::MarkupPerTensorQuantization>(quantizationRestrictions);
    markup->register_pass<ngraph::pass::low_precision::MarkupAvgPoolPrecisionPreserved>();
    markup->register_pass<ngraph::pass::low_precision::PropagatePrecisions>();
    markup->register_pass<ngraph::pass::low_precision::AlignQuantizationIntervals>();
    markup->register_pass<ngraph::pass::low_precision::AlignQuantizationParameters>();

    decompose = std::make_shared<ngraph::pass::Manager>();

    // TODO: to debug only
//    {
//        ngraph::pass::Manager tmp;
//        tmp.register_pass<ngraph::pass::low_precision::MarkupPrecisions>(supportedPrecisions);
//        tmp.run_passes(actualFunction);
//        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/cpu.transforming1.svg").run_on_function(actualFunction);
//        //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\cpu.transforming1").run_on_function(f);
//    }
//
//    {
//        ngraph::pass::Manager tmp;
//        tmp.register_pass<ngraph::pass::low_precision::MarkupPerTensorQuantization>();
//        tmp.run_passes(actualFunction);
//        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/cpu.transforming2.svg").run_on_function(actualFunction);
//        //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\cpu.transforming1").run_on_function(f);
//    }
//
//    {
//        ngraph::pass::Manager tmp;
//        tmp.register_pass<ngraph::pass::low_precision::MarkupAvgPoolPrecisionPreserved>();
//        tmp.run_passes(actualFunction);
//        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/cpu.transforming3.svg").run_on_function(actualFunction);
//        //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\cpu.transforming2").run_on_function(f);
//    }
//
//    {
//        ngraph::pass::Manager tmp;
//        tmp.register_pass<ngraph::pass::low_precision::PropagatePrecisions>();
//        tmp.run_passes(actualFunction);
//        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/cpu.transforming4.svg").run_on_function(actualFunction);
//        //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\cpu.transforming3").run_on_function(f);
//    }
//
//    {
//        ngraph::pass::Manager tmp;
//        tmp.register_pass<ngraph::pass::low_precision::AlignQuantizationIntervals>();
//        tmp.run_passes(actualFunction);
//        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/cpu.transforming5.svg").run_on_function(actualFunction);
//        //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\cpu.transforming4").run_on_function(f);
//    }
//
//    {
//        ngraph::pass::Manager tmp;
//        tmp.register_pass<ngraph::pass::low_precision::AlignQuantizationParameters>();
//        tmp.run_passes(actualFunction);
//        ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/cpu.transforming6.svg").run_on_function(actualFunction);
//        //ngraph::pass::VisualizeTree("c:\\Projects\\temp\\cpu.transforming5").run_on_function(f);
//    }
//
//    {
//        ngraph::pass::Manager tmp;
//        std::shared_ptr<ngraph::pass::GraphRewrite> common = tmp.register_pass<ngraph::pass::GraphRewrite>();
//        common->add_matcher<ngraph::pass::low_precision::PReluTransformation>(params);
//        common->add_matcher<ngraph::pass::low_precision::ConvolutionTransformation>(params);
//        common->add_matcher<ngraph::pass::low_precision::FakeQuantizeDecompositionTransformation>(params);
//        common->add_matcher<ngraph::pass::low_precision::MaxPoolTransformation>(params);
//        tmp.run_passes(actualFunction);
//    }
//
//    ngraph::pass::VisualizeTree("/Users/eshoguli/projects/temp/test.transformed.svg").run_on_function(actualFunction);

    common = std::make_shared<ngraph::pass::Manager>();
    commonGraphRewrite = common->register_pass<ngraph::pass::GraphRewrite>();
}

void SimpleLowPrecisionTransformer::transform(std::shared_ptr<ngraph::Function>& function) {
    ngraph::pass::low_precision::LowPrecision::TypeRelaxedReplacer pass;
    pass.run_on_function(function);

    markup->run_passes(function);
    decompose->run_passes(function);
    common->run_passes(function);
}
