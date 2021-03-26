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

using PrecisionsAttribute = std::set<ngraph::element::Type>;

extern template class TRANSFORMATIONS_API ngraph::VariantImpl<PrecisionsAttribute>;

template<>
class TRANSFORMATIONS_API ngraph::VariantWrapper<PrecisionsAttribute> : public ngraph::VariantImpl<PrecisionsAttribute> {
public:
    static constexpr ngraph::VariantTypeInfo type_info{ "PRECISIONS", 0 };

    const ngraph::VariantTypeInfo& get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}

    std::shared_ptr<ngraph::Variant> merge(const ngraph::NodeVector& nodes) override;

    std::shared_ptr<ngraph::Variant> init(const std::shared_ptr<ngraph::Node>& node) override;

    PrecisionsAttribute get() { return this->m_value; };

    virtual std::string get_string();
};
