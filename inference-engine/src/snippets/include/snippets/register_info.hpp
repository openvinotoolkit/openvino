// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/axis_vector.hpp>

namespace ngraph {

template <>
class TRANSFORMATIONS_API VariantWrapper<std::vector<size_t>> : public VariantImpl<std::vector<size_t>> {
public:
    static constexpr VariantTypeInfo type_info{"Variant::RegInfo|Variant::RuntimeAttribute::AxisVector", 0};

    const VariantTypeInfo& get_type_info() const override { return type_info; }
    VariantWrapper(const value_type& value)
        : VariantImpl<value_type>(value) {
    }
};

} // namespace ngraph