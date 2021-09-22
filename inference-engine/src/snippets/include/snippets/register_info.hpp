// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/axis_vector.hpp>
#include <ngraph/variant.hpp>
#include <transformations_visibility.hpp>

namespace ov {

template <>
class TRANSFORMATIONS_API VariantWrapper<std::vector<size_t>> : public VariantImpl<std::vector<size_t>> {
public:
    static constexpr VariantTypeInfo type_info{"Variant::RegInfo|Variant::RuntimeAttribute::AxisVector", 0};

    const VariantTypeInfo& get_type_info() const override {
        return type_info;
    }
    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}
};

}  // namespace ov
