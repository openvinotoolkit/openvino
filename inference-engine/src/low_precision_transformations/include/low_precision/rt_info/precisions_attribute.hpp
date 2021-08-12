// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/variant.hpp>

#include "low_precision/layer_transformation.hpp"
#include "low_precision/lpt_visibility.hpp"
#include "low_precision/rt_info/attribute_parameters.hpp"
#include "low_precision/rt_info/shared_value_attribute.hpp"

namespace ngraph {

class PrecisionsAttribute;

class LP_TRANSFORMATIONS_API PrecisionsSharedValue : public SharedValue<PrecisionsAttribute> {
public:
    std::vector<ngraph::element::Type> precisions;
};

using PrecisionsAttributePtr = std::shared_ptr<PrecisionsAttribute>;

class LP_TRANSFORMATIONS_API PrecisionsAttribute : public SharedValueAttribute<PrecisionsSharedValue> {
public:
    static const std::vector<ngraph::element::Type> defaultPrecisions;
    PrecisionsAttribute(const std::vector<ngraph::element::Type>& precisions = defaultPrecisions);
};

extern template class LP_TRANSFORMATIONS_API ngraph::VariantImpl<std::shared_ptr<PrecisionsAttribute>>;

template<>
class LP_TRANSFORMATIONS_API VariantWrapper<std::shared_ptr<PrecisionsAttribute>> : public VariantImpl<std::shared_ptr<PrecisionsAttribute>> {
public:
    static constexpr VariantTypeInfo type_info{ "LowPrecision::Precisions", 0 };

    const VariantTypeInfo& get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}

    std::shared_ptr<Variant> init(const std::shared_ptr<ngraph::Node>& node) override;

    std::shared_ptr<PrecisionsAttribute> get() { return this->m_value; }

    // create attribute instance for node
    static std::shared_ptr<VariantWrapper<std::shared_ptr<PrecisionsAttribute>>> create(
        const std::shared_ptr<ngraph::Node>& node,
        const AttributeParameters& params);
    // merge attribute instances which can be got from different sources: node, input port or output port
    void merge(std::vector<std::shared_ptr<VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>>& attributes);
    // vizualize shared attributes details in VizualizeTree pass
    std::string to_string() override;
};
} // namespace ngraph
