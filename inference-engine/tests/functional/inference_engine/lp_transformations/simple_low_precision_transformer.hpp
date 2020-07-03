// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>

#include <ngraph/ngraph.hpp>

#include "common_test_utils/test_common.hpp"
#include "transformations/low_precision/layer_transformation.hpp"
#include "transformations/low_precision/transformation_context.hpp"
#include <transformations/low_precision/transformer.hpp>
#include <transformations/low_precision/iparams_manager.hpp>
#include <transformations/low_precision/ilayer_transformations_manager.hpp>

class SimpleLowPrecisionTransformer : public
    ngraph::pass::IParamsManager,
    ngraph::pass::ILayerTransformationsManager {
public:
    SimpleLowPrecisionTransformer();

    // IParamsManager interface implementation
    std::vector<ngraph::element::Type> getPrecisionsOnActivations(const ngraph::Node& op) const noexcept override;

    // ILayerTransformationsManager interface implementation
    bool isQuantized(std::shared_ptr<ngraph::Node> layer) const noexcept override;
    bool isPrecisionPreserved(std::shared_ptr<ngraph::Node> layer) const noexcept override;

    template <class T, class Operation>
    ngraph::pass::low_precision::LayerTransformationPtr add(const ngraph::pass::low_precision::LayerTransformation::Params& params) {
        const std::string typeName = typeid(ngraph::op::TypeRelaxed<Operation>).name();

        const auto it = transformations.find(typeName);
        if (it != transformations.end()) {
            transformations.erase(it);
        }

        auto transformation = std::make_shared<T>(params);
        transformations.emplace(typeName, transformation);
        return transformation;
    }

    void transform(std::shared_ptr<ngraph::Function>& function);

private:
    std::map<std::string, ngraph::pass::low_precision::LayerTransformationPtr> transformations;
};
