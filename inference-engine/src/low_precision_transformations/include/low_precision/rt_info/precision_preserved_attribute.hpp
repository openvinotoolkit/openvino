// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include "low_precision/lpt_visibility.hpp"
#include "low_precision/rt_info/shared_value_attribute.hpp"

namespace ngraph {

class LP_TRANSFORMATIONS_API PrecisionPreservedAttribute;

class LP_TRANSFORMATIONS_API PrecisionPreservedSharedValue : public SharedValue<PrecisionPreservedAttribute> {
public:
    PrecisionPreservedSharedValue() = default;
    PrecisionPreservedSharedValue(const bool value) : value(value) {}
    bool value;
};

class LP_TRANSFORMATIONS_API PrecisionPreservedAttribute : public SharedValueAttribute<PrecisionPreservedSharedValue> {
public:
    PrecisionPreservedAttribute() = default;
    PrecisionPreservedAttribute(const bool value);
};

using PrecisionPreservedAttributePtr = std::shared_ptr<PrecisionPreservedAttribute>;

extern template class LP_TRANSFORMATIONS_API ngraph::VariantImpl<PrecisionPreservedAttributePtr>;

template<>
class LP_TRANSFORMATIONS_API VariantWrapper<PrecisionPreservedAttributePtr> : public VariantImpl<PrecisionPreservedAttributePtr> {
public:
    static constexpr VariantTypeInfo type_info{ "LowPrecision::PrecisionPreserved", 0 };

    const VariantTypeInfo& get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}

    PrecisionPreservedAttributePtr get() { return this->m_value; }

    std::string to_string() override;
};
} // namespace ngraph
