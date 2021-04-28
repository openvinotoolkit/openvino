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
#include "low_precision/rt_info/precision_preserved_attribute.hpp"

// TODO: not completed
class AvgPoolPrecisionPreservedAttribute {
public:
    AvgPoolPrecisionPreservedAttribute(std::shared_ptr<PrecisionPreservedAttribute::SharedValue> value) : precisionPreservedValue(value) {}

    template <class Operation>
    static std::shared_ptr<AvgPoolPrecisionPreservedAttribute> create(const bool value) {
        // TODO: do we need operation version here?
        auto operationName = Operation::get_type_info_static().name;
        return std::make_shared<AvgPoolPrecisionPreservedAttribute>(value, operationName);
    }

    // TODO: not completed: should be vector to store several shared values, but it's not SharedValueAttribute
    std::shared_ptr<PrecisionPreservedAttribute::SharedValue> precisionPreservedValue;
};

using AvgPoolPrecisionPreservedAttributePtr = std::shared_ptr<AvgPoolPrecisionPreservedAttribute>;

extern template class TRANSFORMATIONS_API ngraph::VariantImpl<AvgPoolPrecisionPreservedAttributePtr>;

template<>
class TRANSFORMATIONS_API ngraph::VariantWrapper<AvgPoolPrecisionPreservedAttributePtr> : public ngraph::VariantImpl<AvgPoolPrecisionPreservedAttributePtr> {
public:
    static constexpr ngraph::VariantTypeInfo type_info{ "LowPrecision::AvgPoolPrecisionPreserved", 0 };

    const ngraph::VariantTypeInfo& get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}

    std::shared_ptr<ngraph::Variant> merge(const ngraph::NodeVector& nodes) override;

    // TODO: new method: need this method to merge attribute instances which can be got from different sources: node/input port/output port
    void merge(std::vector<std::shared_ptr<VariantWrapper<AvgPoolPrecisionPreservedAttributePtr>>>& attributes) {}

    AvgPoolPrecisionPreservedAttributePtr get() { return this->m_value; }

    std::string get_string() override;
};
