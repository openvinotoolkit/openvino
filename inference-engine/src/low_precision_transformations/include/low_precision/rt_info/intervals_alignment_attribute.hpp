// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include <low_precision/lpt_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "low_precision/rt_info/shared_value_attribute.hpp"
#include "attribute_parameters.hpp"

class IntervalsAlignmentAttribute;

class LP_TRANSFORMATIONS_API IntervalsAlignmentSharedValue : public SharedValue<IntervalsAlignmentAttribute> {
public:
    IntervalsAlignmentSharedValue() = default;
    IntervalsAlignmentSharedValue(const float intervalLow, const float intervalHigh, const bool isValid = true) :
            intervalLow(intervalLow), intervalHigh(intervalHigh), isValid(isValid) {}
    float intervalLow;
    float intervalHigh;
    bool isValid;
};

class LP_TRANSFORMATIONS_API IntervalsAlignmentAttribute : public SharedValueAttribute<IntervalsAlignmentSharedValue> {
public:
    IntervalsAlignmentAttribute() = default;
    IntervalsAlignmentAttribute(const float intervalLow, const float intervalHigh, const bool isValid = true);
};

using IntervalsAlignmentAttributePtr = std::shared_ptr<IntervalsAlignmentAttribute>;

extern template class LP_TRANSFORMATIONS_API ngraph::VariantImpl<IntervalsAlignmentAttributePtr>;

template<>
class LP_TRANSFORMATIONS_API ngraph::VariantWrapper<std::shared_ptr<IntervalsAlignmentAttribute>> :
    public ngraph::VariantImpl<std::shared_ptr<IntervalsAlignmentAttribute>> {
public:
    static constexpr ngraph::VariantTypeInfo type_info{ "LowPrecision::IntervalsAlignment", 0 };

    const ngraph::VariantTypeInfo& get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}

    std::shared_ptr<ngraph::Variant> merge(const ngraph::NodeVector& nodes) override;

    std::shared_ptr<IntervalsAlignmentAttribute> get() const { return this->m_value; }

    static std::shared_ptr<VariantWrapper<std::shared_ptr<IntervalsAlignmentAttribute>>> create(
        const std::shared_ptr<ngraph::Node>& node,
        const AttributeParameters& params);
    void merge(std::vector<std::shared_ptr<VariantWrapper<std::shared_ptr<IntervalsAlignmentAttribute>>>>& attributes);
    std::string get_string() override;
};
