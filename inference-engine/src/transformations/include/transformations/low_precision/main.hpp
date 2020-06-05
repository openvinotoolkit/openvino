// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ie_api.h>

#include <ngraph/ngraph.hpp>

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations/low_precision/ilayer_transformations_manager.hpp>
#include <transformations/low_precision/iparams_manager.hpp>

using namespace std;


namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API LowPrecisionTransformations: public ngraph::pass::GraphRewrite, IParamsManager, ILayerTransformationsManager {
public:
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

    // IParamsManager interface implementation
    std::vector<element::Type> getPrecisionsOnActivations(const NodeTypeInfo& layerName) const noexcept override;

    // ILayerTransformationsManager interface implementation
    bool isQuantized(std::shared_ptr<Node> layer) const noexcept override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
};

}// namespace pass
}// namespace ngraph
