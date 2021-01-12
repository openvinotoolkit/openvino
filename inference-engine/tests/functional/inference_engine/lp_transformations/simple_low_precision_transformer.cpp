// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "simple_low_precision_transformer.hpp"

#include <string>
#include <ngraph/ngraph.hpp>
#include <low_precision/transformation_context.hpp>
#include <low_precision/transformer.hpp>
#include <low_precision/layer_transformation.hpp>
#include <low_precision/transformation_context.hpp>

#include <low_precision/markup_precisions.hpp>
#include <low_precision/markup_avg_pool_precisions.hpp>
#include <low_precision/propagate_precisions.hpp>
#include <low_precision/align_concat_quantization_parameters.hpp>

using namespace testing;
using namespace ngraph::pass;

SimpleLowPrecisionTransformer::SimpleLowPrecisionTransformer() {
    auto passConfig = std::make_shared<PassConfig>();
    lowPrecisionManager = std::make_shared<ngraph::pass::Manager>();
    lowPrecisionManager->register_pass<ngraph::pass::low_precision::MarkupPrecisions>();
    lowPrecisionManager->register_pass<ngraph::pass::low_precision::MarkupAvgPoolPrecisions>();
    lowPrecisionManager->register_pass<ngraph::pass::low_precision::PropagatePrecisions>();
    lowPrecisionManager->register_pass<ngraph::pass::low_precision::AlignConcatQuantizationParamters>();
}

std::vector<ngraph::element::Type> SimpleLowPrecisionTransformer::getPrecisionsOnActivations(const ngraph::Node& op) const noexcept {
    const auto it = transformations.find(ngraph::pass::low_precision::LowPrecisionTransformations::getType(op));
    if (it == transformations.end()) {
        return std::vector<ngraph::element::Type>();
    }

    const ngraph::pass::low_precision::LayerTransformationPtr transformation = it->second;
    return transformation->getPrecisionsOnActivations();
}

bool SimpleLowPrecisionTransformer::isQuantized(const std::shared_ptr<ngraph::Node>& layer) const noexcept {
    const std::string operantionType = ngraph::pass::low_precision::LowPrecisionTransformations::getType(*layer);

    const auto it = transformations.find(operantionType);
    if (it == transformations.end()) {
        return false;
    }

    const ngraph::pass::low_precision::LayerTransformationPtr transformation = it->second;
    return transformation->isQuantized(layer);
}

bool SimpleLowPrecisionTransformer::isPrecisionPreserved(const std::shared_ptr<ngraph::Node>& layer) const noexcept {
    const std::string operantionType = ngraph::pass::low_precision::LowPrecisionTransformations::getType(*layer);

    const auto it = transformations.find(operantionType);
    if (it == transformations.end()) {
        return false;
    }

    const ngraph::pass::low_precision::LayerTransformationPtr transformation = it->second;
    return transformation->isPrecisionPreserved(layer);
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
    context.function = function;
    lowPrecisionManager->run_passes(function);
}
