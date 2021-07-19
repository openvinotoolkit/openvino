// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include "low_precision/lpt_visibility.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"

namespace ngraph {
class LP_TRANSFORMATIONS_API AvgPoolPrecisionPreservedAttribute : public PrecisionPreservedAttribute {
};

using AvgPoolPrecisionPreservedAttributePtr = std::shared_ptr<AvgPoolPrecisionPreservedAttribute>;

extern template class LP_TRANSFORMATIONS_API VariantImpl<AvgPoolPrecisionPreservedAttributePtr>;

template<>
class LP_TRANSFORMATIONS_API VariantWrapper<AvgPoolPrecisionPreservedAttributePtr> : public VariantImpl<AvgPoolPrecisionPreservedAttributePtr> {
public:
    static constexpr VariantTypeInfo type_info{ "LowPrecision::AvgPoolPrecisionPreserved", 0 };

    const VariantTypeInfo& get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}

    AvgPoolPrecisionPreservedAttributePtr get() { return this->m_value; }

    void merge(std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<AvgPoolPrecisionPreservedAttribute>>>>& attributes);
    std::string to_string() override;
};
} // namespace ngraph
