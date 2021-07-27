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

#include <low_precision/lpt_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "shared_value_attribute.hpp"
#include "attribute_parameters.hpp"

namespace ngraph {
class QuantizationAlignmentAttribute;

class LP_TRANSFORMATIONS_API QuantizationAlignmentSharedValue : public SharedValue<QuantizationAlignmentAttribute> {
public:
    QuantizationAlignmentSharedValue(const bool value = false) : value(value) {}
    bool value;
};

class LP_TRANSFORMATIONS_API QuantizationAlignmentAttribute : public SharedValueAttribute<QuantizationAlignmentSharedValue>{
public:
    QuantizationAlignmentAttribute(const bool value = false);
};

using QuantizationAlignmentAttributePtr = std::shared_ptr<QuantizationAlignmentAttribute>;

extern template class LP_TRANSFORMATIONS_API ngraph::VariantImpl<QuantizationAlignmentAttributePtr>;

template<>
class LP_TRANSFORMATIONS_API VariantWrapper<std::shared_ptr<QuantizationAlignmentAttribute>> :
    public VariantImpl<std::shared_ptr<QuantizationAlignmentAttribute>> {
public:
    static constexpr VariantTypeInfo type_info{ "LowPrecision::QuantizationAlignment", 0 };

    const VariantTypeInfo& get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}

    std::shared_ptr<Variant> init(const std::shared_ptr<ngraph::Node>& node) override;

    std::shared_ptr<QuantizationAlignmentAttribute> get() { return this->m_value; }

    static std::shared_ptr<VariantWrapper<std::shared_ptr<QuantizationAlignmentAttribute>>> create(
        const std::shared_ptr<ngraph::Node>& node,
        const AttributeParameters& params);
    void merge(std::vector<std::shared_ptr<VariantWrapper<std::shared_ptr<QuantizationAlignmentAttribute>>>>& attributes);
    std::string to_string() override;
};
} // namespace ngraph
