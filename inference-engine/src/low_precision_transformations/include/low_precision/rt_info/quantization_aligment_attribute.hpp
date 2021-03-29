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
    class SharedPart {
    public:
        class SharedValue {
        public:
            SharedValue(const float intervalLow, const float intervalHigh) : intervalLow(intervalLow), intervalHigh(intervalHigh) {}
            float intervalLow;
            float intervalHigh;
        };

        SharedPart(std::shared_ptr<SharedValue> value) : value(value) {}
        SharedPart(const float intervalLow, const float intervalHigh) : value(std::make_shared<SharedValue>(intervalLow, intervalHigh)) {}
        std::shared_ptr<SharedValue> value;
    };

    QuantizationAligmentAttribute() {}
    QuantizationAligmentAttribute(const float intervalLow, const float intervalHigh) :
        sharedPart(std::make_shared<SharedPart>(intervalLow, intervalHigh)) {}
    QuantizationAligmentAttribute(std::shared_ptr<QuantizationAligmentAttribute::SharedPart::SharedValue> value) :
        sharedPart(std::make_shared<SharedPart>(value)) {}
    QuantizationAligmentAttribute(std::shared_ptr<SharedPart> sharedPart) : sharedPart(sharedPart) {}

    template <class Operation>
    static QuantizationAligmentAttribute create(const float intervalLow, const float intervalHigh) {
        auto operationName = Operation::get_type_info_static().name;
        return QuantizationAligmentAttribute(intervalLow, intervalHigh);
    }

    std::shared_ptr<SharedPart> sharedPart;
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
