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
        bool value;
    };

    PrecisionPreservedAttribute(const bool value) : sharedValue(std::make_shared<SharedValue>(value)) {}
    PrecisionPreservedAttribute(std::shared_ptr<SharedValue> value) : sharedValue(value) {}

    std::shared_ptr<SharedValue> sharedValue;
};

extern template class TRANSFORMATIONS_API ngraph::VariantImpl<PrecisionPreservedAttribute>;

template<>
class TRANSFORMATIONS_API ngraph::VariantWrapper<PrecisionPreservedAttribute> : public ngraph::VariantImpl<PrecisionPreservedAttribute> {
public:
    static constexpr ngraph::VariantTypeInfo type_info{ "LowPrecision::PrecisionPreserved", 0 };

    const ngraph::VariantTypeInfo& get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}

    PrecisionPreservedAttribute get() { return this->m_value; }

    std::string get_string() override;
};
