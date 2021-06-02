// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>

#include <ngraph/ngraph.hpp>

#include "layer_transformation.hpp"
#include "common_test_utils/test_common.hpp"
#include "low_precision/layer_transformation.hpp"
#include "low_precision/common/operation_precision_restriction.hpp"
#include "low_precision/common/operation_per_tensor_quantization_restriction.hpp"

class SimpleLowPrecisionTransformer {
public:
    SimpleLowPrecisionTransformer(
        const std::vector<ngraph::pass::low_precision::OperationPrecisionRestriction>& precisionRestrictions = {},
        const std::vector<ngraph::pass::low_precision::OperationPerTensorQuantizationRestriction>& quantizationRestrictions = {});

    template <class T, class Operation>
    void add(const TestTransformationParams& params) {
        this->common->add_matcher<T>(TestTransformationParams::toParams(params));
    }

    template <class T>
    void register_pass() {
        lowPrecisionManager->register_pass<T>();
    }

    template <class T>
    void set_callback(const std::function<bool(const std::shared_ptr<const ::ngraph::Node>)>& callback) {
        lowPrecisionManager->get_pass_config()->set_callback<T>(callback);
    }

    void transform(std::shared_ptr<ngraph::Function>& function);

private:
    ngraph::pass::low_precision::TransformationContext context;
    std::shared_ptr<ngraph::pass::Manager> lowPrecisionManager;
    std::shared_ptr<ngraph::pass::GraphRewrite> common;
    std::map<std::string, ngraph::pass::low_precision::LayerTransformationPtr> transformations;
};
