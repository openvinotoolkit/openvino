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

class QuantizationAligmentAttribute {
public:
    class SharedValue {
    public:
        SharedValue(const float intervalLow, const float intervalHigh, const size_t levels) :
            intervalLow(intervalLow),
            intervalHigh(intervalHigh),
            levels(levels) {}

        float intervalLow;
        float intervalHigh;
        size_t levels;
    };

    QuantizationAligmentAttribute(const float intervalLow, const float intervalHigh, const size_t levels) :
        sharedValue(std::make_shared<SharedValue>(intervalLow, intervalHigh, levels)) {}
    QuantizationAligmentAttribute(std::shared_ptr<SharedValue> sharedValue) : sharedValue(sharedValue) {}

    template <class Operation>
    static QuantizationAligmentAttribute create(const bool value) {
        // TODO: do we need operation version here?
        auto operationName = Operation::get_type_info_static().name;
        return PrecisionPreservedAttribute(value, operationName);
    }

    std::shared_ptr<SharedValue> sharedValue;
};

extern template class TRANSFORMATIONS_API ngraph::VariantImpl<QuantizationAligmentAttribute>;

template<>
class TRANSFORMATIONS_API ngraph::VariantWrapper<QuantizationAligmentAttribute> : public ngraph::VariantImpl<QuantizationAligmentAttribute> {
public:
    static constexpr ngraph::VariantTypeInfo type_info{ "QUANTIZATION_ALIGMENT", 0 };

    const ngraph::VariantTypeInfo& get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}

    // TODO: not completed for several branches
    std::shared_ptr<ngraph::Variant> merge(const ngraph::NodeVector& nodes) override;

    std::shared_ptr<ngraph::Variant> init(const std::shared_ptr<ngraph::Node>& node) override;

    QuantizationAligmentAttribute get() { return this->m_value; };

    std::string get_string() override;
};
