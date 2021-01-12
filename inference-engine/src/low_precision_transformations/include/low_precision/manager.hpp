// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "iparams_manager.hpp"
#include "ilayer_transformations_manager.hpp"
#include "low_precision/layer_transformation.hpp"
#include "low_precision/transformation_context.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API Manager;

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph

class ngraph::pass::low_precision::Manager: public ngraph::pass::Manager, IParamsManager, public ILayerTransformationsManager {
public:
    Manager();
    explicit Manager(
        std::shared_ptr<PassConfig> pass_config,
        TransformationContext* context,
        const ILayerTransformationsManager* manager = nullptr);

    template <typename T, typename Operation, bool Enable = true, class... Args>
    std::shared_ptr<T> register_pass(Args&&... args)
    {
        static_assert(std::is_base_of<ngraph::pass::low_precision::LayerTransformation, T>::value, "not derived from LPT base type");

        auto rc = ngraph::pass::Manager::register_pass<T>(std::forward<Args>(args)...);
        auto transformation = std::dynamic_pointer_cast<ngraph::pass::low_precision::LayerTransformation>(rc);

        transformation->setContext(context);
        transformation->setParamsManager(this);
        transformation->setLayerTransformationsManager(this);
        transformations.emplace(Operation::get_type_info_static().name, transformation);

        return rc;
    }

    // IParamsManager interface implementation
    std::vector<element::Type> getPrecisionsOnActivations(const Node& op) const noexcept override;

    // ILayerTransformationsManager interface implementation
    bool isQuantized(const std::shared_ptr<Node>& layer) const noexcept override;
    bool isPrecisionPreserved(const std::shared_ptr<Node>& layer) const noexcept override;

    // Find transformations which can handle operation by operation name.
    // Note, there are no dequantization operations are sill here, as result not possible to use transformation matcher->match(target)
    std::vector<LayerTransformationPtr> find(const std::string& name) const noexcept;

protected:
    TransformationContext* context;
    const ILayerTransformationsManager* manager;
    std::map<std::string, LayerTransformationPtr> transformations;
};
