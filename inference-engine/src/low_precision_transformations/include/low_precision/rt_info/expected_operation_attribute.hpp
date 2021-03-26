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

class ExpectedOperationAttribute {
public:
    class SharedValue {
    public:
        // SharedValue() : value(false) /*, empty(true) */ {}
        SharedValue(const bool value) : value(value) /*, empty(true) */ {}
        SharedValue(const bool value, const std::string& operationName) : value(value), operationName(operationName) /*, empty(false)*/ {}
        std::string operationName;
        bool value;
        //bool empty;
    };

    // ExpectedOperationAttribute() {}
    ExpectedOperationAttribute(const bool value, const std::string& operationName) : sharedValue(std::make_shared<SharedValue>(value, operationName)) {}
    ExpectedOperationAttribute(const bool value) : sharedValue(std::make_shared<SharedValue>(value)) {}
    ExpectedOperationAttribute(std::shared_ptr<SharedValue> sharedValue) : sharedValue(sharedValue) {}

    template <class Operation>
    static ExpectedOperationAttribute create(const bool value) {
        // TODO: do we need operation version here?
        auto operationName = Operation::get_type_info_static().name;
        return ExpectedOperationAttribute(value, operationName);
    }

    std::shared_ptr<SharedValue> sharedValue;
};

extern template class TRANSFORMATIONS_API ngraph::VariantImpl<ExpectedOperationAttribute>;

template<>
class TRANSFORMATIONS_API ngraph::VariantWrapper<ExpectedOperationAttribute> : public ngraph::VariantImpl<ExpectedOperationAttribute> {
public:
    static constexpr ngraph::VariantTypeInfo type_info{ "PRECISION_PRESERVED", 0 };

    const ngraph::VariantTypeInfo& get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}

    // TODO: not completed for several branches
    std::shared_ptr<ngraph::Variant> merge(const ngraph::NodeVector& nodes) override;

    std::shared_ptr<ngraph::Variant> init(const std::shared_ptr<ngraph::Node>& node) override;

    ExpectedOperationAttribute get() { return this->m_value; };

    std::string get_string() override;
};
