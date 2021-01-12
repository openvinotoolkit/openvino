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

class PrecisionsAttribute {
public:
    PrecisionsAttribute(const std::set<ngraph::element::Type>& precisions) : precisions(precisions) {}
    std::set<ngraph::element::Type> precisions;
};

extern template class TRANSFORMATIONS_API ngraph::VariantImpl<std::shared_ptr<PrecisionsAttribute>>;

template<>
class TRANSFORMATIONS_API ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>> : public ngraph::VariantImpl<std::shared_ptr<PrecisionsAttribute>> {
public:
    static constexpr ngraph::VariantTypeInfo type_info{ "PRECISIONS", 0 };

    const ngraph::VariantTypeInfo& get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}

    std::shared_ptr<ngraph::Variant> merge(const ngraph::NodeVector& nodes) override;

    void merge(std::vector<std::shared_ptr<VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>>& attributes);

    std::shared_ptr<ngraph::Variant> init(const std::shared_ptr<ngraph::Node>& node) override;

    std::shared_ptr<PrecisionsAttribute> get() { return this->m_value; };

    virtual std::string get_string();
};
