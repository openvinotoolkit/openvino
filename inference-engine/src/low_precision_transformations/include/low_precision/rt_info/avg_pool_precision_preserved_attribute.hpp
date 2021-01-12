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

#include <low_precision/lpt_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/precisions_attribute.hpp"
#include "low_precision/rt_info/precision_preserved_attribute.hpp"
#include "low_precision/rt_info/shared_value_attribute.hpp"

class LP_TRANSFORMATIONS_API AvgPoolPrecisionPreservedAttribute : public PrecisionPreservedAttribute {
public:
};

using AvgPoolPrecisionPreservedAttributePtr = std::shared_ptr<AvgPoolPrecisionPreservedAttribute>;

extern template class LP_TRANSFORMATIONS_API ngraph::VariantImpl<AvgPoolPrecisionPreservedAttributePtr>;

template<>
class LP_TRANSFORMATIONS_API ngraph::VariantWrapper<AvgPoolPrecisionPreservedAttributePtr> : public ngraph::VariantImpl<AvgPoolPrecisionPreservedAttributePtr> {
public:
    static constexpr ngraph::VariantTypeInfo type_info{ "LowPrecision::AvgPoolPrecisionPreserved", 0 };

    const ngraph::VariantTypeInfo& get_type_info() const override {
        return type_info;
    }

    VariantWrapper(const value_type& value) : VariantImpl<value_type>(value) {}

    std::shared_ptr<ngraph::Variant> merge(const ngraph::NodeVector& nodes) override;

    AvgPoolPrecisionPreservedAttributePtr get() { return this->m_value; }

    // TODO: new method: need this method to merge attribute instances which can be got from different sources: node/input port/output port
    void merge(std::vector<std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<AvgPoolPrecisionPreservedAttribute>>>>& attributes);
    std::string get_string() override;
};
