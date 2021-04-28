// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "low_precision/rt_info/shared_value_attribute.hpp"

class PrecisionsAttribute;

class TRANSFORMATIONS_API PrecisionsSharedValue : public SharedValue<PrecisionsAttribute> {
public:
    std::set<ngraph::element::Type> precisions;
};

class TRANSFORMATIONS_API PrecisionsAttribute : public SharedValueAttribute<PrecisionsSharedValue> {
public:
    PrecisionsAttribute(const std::set<ngraph::element::Type>& precisions = std::set<ngraph::element::Type>{ ngraph::element::u8, ngraph::element::i8 });
};

extern template class TRANSFORMATIONS_API ngraph::VariantImpl<std::shared_ptr<PrecisionsAttribute>>;

template<>
class TRANSFORMATIONS_API ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>> : public ngraph::VariantImpl<std::shared_ptr<PrecisionsAttribute>> {
public:
    static constexpr ngraph::VariantTypeInfo type_info { "LowPrecision::Precisions", 0 };

    const ngraph::VariantTypeInfo& get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}

    std::shared_ptr<ngraph::Variant> merge(const ngraph::NodeVector& nodes) override;

    // TODO: new method: need this method to merge attribute instances which can be got from different sources: node/input port/output port
    void merge(std::vector<std::shared_ptr<VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>>& attributes);

    std::shared_ptr<ngraph::Variant> init(const std::shared_ptr<ngraph::Node>& node) override;

    std::shared_ptr<PrecisionsAttribute> get() { return this->m_value; }

    std::string get_string() override;
};
