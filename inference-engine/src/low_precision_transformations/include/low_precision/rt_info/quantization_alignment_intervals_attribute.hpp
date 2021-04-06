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

class QuantizationAlignmentIntervalsAttribute {
public:
    QuantizationAlignmentIntervalsAttribute(const float intervalLow, const float intervalHigh) : intervalLow(intervalLow), intervalHigh(intervalHigh) {}
    float intervalLow;
    float intervalHigh;
};

using QuantizationAlignmentIntervalsAttributePtr = std::shared_ptr<QuantizationAlignmentIntervalsAttribute>;

extern template class TRANSFORMATIONS_API ngraph::VariantImpl<QuantizationAlignmentIntervalsAttributePtr>;

template<>
class TRANSFORMATIONS_API ngraph::VariantWrapper<std::shared_ptr<QuantizationAlignmentIntervalsAttribute>> :
    public ngraph::VariantImpl<std::shared_ptr<QuantizationAlignmentIntervalsAttribute>> {
public:
    static constexpr ngraph::VariantTypeInfo type_info{ "INTERVALS_ALIGNMENT", 0 };

    const ngraph::VariantTypeInfo& get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}

    std::shared_ptr<ngraph::Variant> merge(const ngraph::NodeVector& nodes) override;

    std::shared_ptr<ngraph::Variant> init(const std::shared_ptr<ngraph::Node>& node) override;

    std::shared_ptr<QuantizationAlignmentIntervalsAttribute> get() { return this->m_value; };

    std::string get_string() override;
};
