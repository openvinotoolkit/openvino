// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>

#include <low_precision/lpt_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "low_precision/rt_info/shared_value_attribute.hpp"
#include "low_precision/layer_transformation.hpp"
#include "attribute_parameters.hpp"

class LP_TRANSFORMATIONS_API PerTensorQuantizationAttribute {
};

extern template class LP_TRANSFORMATIONS_API ngraph::VariantImpl<PerTensorQuantizationAttribute>;

template<>
class LP_TRANSFORMATIONS_API ngraph::VariantWrapper<PerTensorQuantizationAttribute> : public ngraph::VariantImpl<PerTensorQuantizationAttribute> {
public:
    static constexpr ngraph::VariantTypeInfo type_info { "LowPrecision::PerTensorQuantization", 0 };

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}

    const ngraph::VariantTypeInfo& get_type_info() const override {
        return type_info;
    }
};
