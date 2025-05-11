// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>

#include "openvino/pass/manager.hpp"

#include "layer_transformation.hpp"
#include "common_test_utils/test_common.hpp"
#include "low_precision/layer_transformation.hpp"
#include "low_precision/common/precisions_restriction.hpp"
#include "low_precision/common/quantization_granularity_restriction.hpp"

class SimpleLowPrecisionTransformer : public ov::pass::ModelPass{
public:
    OPENVINO_MODEL_PASS_RTTI("SimpleLowPrecisionTransformer");
    SimpleLowPrecisionTransformer(
        const std::vector<ov::pass::low_precision::PrecisionsRestriction>& precisionRestrictions = {},
        const std::vector<ov::pass::low_precision::QuantizationGranularityRestriction>& quantizationRestrictions = {},
        const AttributeParameters& params = AttributeParameters(),
        const bool addCleanup = false);

    template <class T, class Operation>
    void add(const TestTransformationParams& params) {
        commonGraphRewrite->add_matcher<T>(TestTransformationParams::toParams(params));
    }
    template <class T, class Operation>
    void add(const std::shared_ptr<ov::Model> model, const TestTransformationParams& params) {
        commonGraphRewrite->add_matcher<T>(model, TestTransformationParams::toParams(params));
    }
    template <class T>
    void add(const TestTransformationParams& params) {
        commonGraphRewrite->add_matcher<T>(TestTransformationParams::toParams(params));
    }

    void transform(std::shared_ptr<ov::Model>& model);
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

    std::shared_ptr<ov::pass::Manager> markup;
    std::shared_ptr<ov::pass::Manager> common;
    std::shared_ptr<ov::pass::GraphRewrite> commonGraphRewrite;
    std::shared_ptr<ov::pass::GraphRewrite> cleanup;
};
