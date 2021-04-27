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

class QuantizationAlignmentAttribute {
public:
    QuantizationAlignmentAttribute(const bool hasToBeAligned = false) : hasToBeAligned(hasToBeAligned) {}
    bool hasToBeAligned;
};

using QuantizationAlignmentAttributePtr = std::shared_ptr<QuantizationAlignmentAttribute>;

extern template class TRANSFORMATIONS_API ngraph::VariantImpl<QuantizationAlignmentAttributePtr>;

template<>
class TRANSFORMATIONS_API ngraph::VariantWrapper<std::shared_ptr<QuantizationAlignmentAttribute>> :
    public ngraph::VariantImpl<std::shared_ptr<QuantizationAlignmentAttribute>> {
public:
    static constexpr ngraph::VariantTypeInfo type_info{ "LowPrecision::QuantizationAlignment", 0 };

    const ngraph::VariantTypeInfo& get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}

    std::shared_ptr<ngraph::Variant> merge(const ngraph::NodeVector& nodes) override;

    std::shared_ptr<ngraph::Variant> init(const std::shared_ptr<ngraph::Node>& node) override;

    std::shared_ptr<QuantizationAlignmentAttribute> get() { return this->m_value; }

    std::string get_string() override;
};
