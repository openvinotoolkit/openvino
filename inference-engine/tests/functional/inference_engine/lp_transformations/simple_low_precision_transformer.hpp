// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>

#include <ngraph/ngraph.hpp>

#include "common_test_utils/test_common.hpp"
#include "low_precision/layer_transformation.hpp"
#include "low_precision/transformation_context.hpp"
#include <low_precision/iparams_manager.hpp>
#include <low_precision/ilayer_transformations_manager.hpp>

class SimpleLowPrecisionTransformer {
public:
    SimpleLowPrecisionTransformer();

    template <class T, class Operation>
    void add(const ngraph::pass::low_precision::LayerTransformation::Params& params) {
        //// const std::string typeName = typeid(ngraph::op::TypeRelaxed<Operation>).name();
        //const std::string typeName = ngraph::pass::low_precision::LowPrecisionTransformations::getType<Operation>();

        //const auto it = transformations.find(typeName);
        //if (it != transformations.end()) {
        //    transformations.erase(it);
        //}

        //auto transformation = std::make_shared<T>(params);
        //transformations.emplace(typeName, transformation);
        //return transformation;

        lowPrecisionManager->register_pass<T>(params);
    }

    template <class T>
    void register_pass() {
        lowPrecisionManager->register_pass<T>();
    }

    void transform(std::shared_ptr<ngraph::Function>& function);

private:
    ngraph::pass::low_precision::TransformationContext context;
    std::shared_ptr<ngraph::pass::Manager> lowPrecisionManager;
    std::map<std::string, ngraph::pass::low_precision::LayerTransformationPtr> transformations;
};
