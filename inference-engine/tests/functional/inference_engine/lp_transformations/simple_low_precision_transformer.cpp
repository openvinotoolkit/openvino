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

using namespace testing;
using namespace ngraph::pass;

SimpleLowPrecisionTransformer::SimpleLowPrecisionTransformer() {
    auto passConfig = std::make_shared<PassConfig>();
    lowPrecisionManager = std::make_shared<ngraph::pass::Manager>();
    lowPrecisionManager->register_pass<ngraph::pass::low_precision::MarkupPrecisions>();
    lowPrecisionManager->register_pass<ngraph::pass::low_precision::MarkupAvgPoolPrecisionPreserved>();
    lowPrecisionManager->register_pass<ngraph::pass::low_precision::PropagatePrecisions>();
    lowPrecisionManager->register_pass<ngraph::pass::low_precision::AlignConcatQuantizationParamters>();
}

void SimpleLowPrecisionTransformer::transform(std::shared_ptr<ngraph::Function>& function) {
    //{
    //    ngraph::pass::low_precision::TypeRelaxedReplacer pass;
    //    pass.run_on_function(function);
    //}

    //ngraph::pass::low_precision::TransformationContext context(function);
    //GraphRewrite pass;
    //for (auto it : transformations) {
    //    ngraph::pass::low_precision::LayerTransformationPtr transformation = it.second;

    //    transformation->setParamsManager(this);
    //    transformation->setLayerTransformationsManager(this);
    //    transformation->registerMatcherIn(pass, context);
    //}
    //pass.run_on_function(function);
    ngraph::pass::low_precision::LowPrecision::TypeRelaxedReplacer pass;
    pass.run_on_function(function);

    context.function = function;
    lowPrecisionManager->run_passes(function);
}
