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

class PrecisionPreservedAttribute {
public:
    class SharedValue {
    public:
        SharedValue(const bool value) : value(value) {}
        SharedValue(const bool value, const std::string& operationName) : value(value), operationName(operationName) {}
        std::string operationName;
        bool value;
    };

    PrecisionPreservedAttribute(const bool value, const std::string& operationName) : sharedValue(std::make_shared<SharedValue>(value, operationName)) {}
    PrecisionPreservedAttribute(const bool value) : sharedValue(std::make_shared<SharedValue>(value)) {}
    PrecisionPreservedAttribute(std::shared_ptr<SharedValue> sharedValue) : sharedValue(sharedValue) {}

    template <class Operation>
    static PrecisionPreservedAttribute create(const bool value) {
        // TODO: do we need operation version here?
        auto operationName = Operation::get_type_info_static().name;
        return PrecisionPreservedAttribute(value, operationName);
    }

    std::shared_ptr<SharedValue> sharedValue;
};

extern template class TRANSFORMATIONS_API ngraph::VariantImpl<PrecisionPreservedAttribute>;

template<>
class TRANSFORMATIONS_API ngraph::VariantWrapper<PrecisionPreservedAttribute> : public ngraph::VariantImpl<PrecisionPreservedAttribute> {
public:
    static constexpr ngraph::VariantTypeInfo type_info{ "PRECISION_PRESERVED", 0 };

    const ngraph::VariantTypeInfo& get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}

    std::shared_ptr<ngraph::Variant> merge(const ngraph::NodeVector& nodes) override;

    PrecisionPreservedAttribute get() { return this->m_value; };

    std::string get_string() override;
};
