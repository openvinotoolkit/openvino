// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>

#include <ngraph/ngraph.hpp>

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
    void add(const ngraph::pass::low_precision::LayerTransformation::Params& params) {
        this->common->add_matcher<T>(params);
    }

    template <class T>
    void register_pass() {
        lowPrecisionManager->register_pass<T>();
    }

    void transform(std::shared_ptr<ngraph::Function>& function);

private:
    ngraph::pass::low_precision::TransformationContext context;
    std::shared_ptr<ngraph::pass::Manager> lowPrecisionManager;
    std::shared_ptr<ngraph::pass::GraphRewrite> common;
    std::map<std::string, ngraph::pass::low_precision::LayerTransformationPtr> transformations;
};
