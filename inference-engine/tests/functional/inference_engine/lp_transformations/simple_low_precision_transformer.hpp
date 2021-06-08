// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>

#include <ngraph/ngraph.hpp>

#include "common_test_utils/test_common.hpp"
#include "low_precision/layer_transformation.hpp"
#include "low_precision/transformation_context.hpp"
#include <low_precision/transformer.hpp>
#include <low_precision/iparams_manager.hpp>
#include <low_precision/ilayer_transformations_manager.hpp>

class SimpleLowPrecisionTransformer : public
    ngraph::pass::IParamsManager,
    ngraph::pass::ILayerTransformationsManager {
public:
    SimpleLowPrecisionTransformer();

    // IParamsManager interface implementation
    std::vector<ngraph::element::Type> getPrecisionsOnActivations(const ngraph::Node& op) const noexcept override;

    // ILayerTransformationsManager interface implementation
    bool isQuantized(const std::shared_ptr<ngraph::Node>& layer) const noexcept override;
    bool isPrecisionPreserved(const std::shared_ptr<ngraph::Node>& layer) const noexcept override;

    template <class T, class Operation>
    ngraph::pass::low_precision::LayerTransformationPtr addBranchSpecific(const ngraph::pass::low_precision::LayerTransformation::Params& params) {
        const std::string typeName = ngraph::pass::low_precision::LowPrecisionTransformations::getType<Operation>();

        const auto it = branchSpecificTransformations.find(typeName);
        if (it != branchSpecificTransformations.end()) {
            branchSpecificTransformations.erase(it);
        }

        auto transformation = std::make_shared<T>(params);
        branchSpecificTransformations.emplace(typeName, transformation);
        return transformation;
    }

    template <class T, class Operation>
    ngraph::pass::low_precision::LayerTransformationPtr add(const ngraph::pass::low_precision::LayerTransformation::Params& params) {
        const std::string typeName = ngraph::pass::low_precision::LowPrecisionTransformations::getType<Operation>();

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
    std::map<std::string, ngraph::pass::low_precision::LayerTransformationPtr> branchSpecificTransformations;
    std::map<std::string, ngraph::pass::low_precision::LayerTransformationPtr> transformations;
};
