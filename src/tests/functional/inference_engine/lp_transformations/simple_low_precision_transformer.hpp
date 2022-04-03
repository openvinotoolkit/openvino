// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>

#include <ngraph/ngraph.hpp>

#include "layer_transformation.hpp"
#include "common_test_utils/test_common.hpp"
#include "low_precision/layer_transformation.hpp"
#include "low_precision/common/precisions_restriction.hpp"
#include "low_precision/common/quantization_granularity_restriction.hpp"

class SimpleLowPrecisionTransformer : public ngraph::pass::FunctionPass{
public:
    SimpleLowPrecisionTransformer(
        const std::vector<ngraph::pass::low_precision::PrecisionsRestriction>& precisionRestrictions = {},
        const std::vector<ngraph::pass::low_precision::QuantizationGranularityRestriction>& quantizationRestrictions = {},
        const AttributeParameters& params = AttributeParameters());

    template <class T, class Operation>
    void add(const TestTransformationParams& params) {
        commonGraphRewrite->add_matcher<T>(TestTransformationParams::toParams(params));
    }
    template <class T, class Operation>
    void add(const std::shared_ptr<ngraph::Function> function, const TestTransformationParams& params) {
        commonGraphRewrite->add_matcher<T>(function, TestTransformationParams::toParams(params));
    }

    void transform(std::shared_ptr<ngraph::Function>& function);
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;

    std::shared_ptr<ngraph::pass::Manager> markup;
    std::shared_ptr<ngraph::pass::Manager> common;
    std::shared_ptr<ngraph::pass::GraphRewrite> commonGraphRewrite;
    std::shared_ptr<ngraph::pass::GraphRewrite> cleanup;
};
