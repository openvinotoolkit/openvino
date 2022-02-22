// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/primitives_priority_attribute.hpp"

#include <assert.h>

#include <functional>
#include <iterator>
#include <memory>
#include <ngraph/node.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/variant.hpp>
#include <ostream>

using namespace ov;
using namespace ngraph;

std::string ov::getPrimitivesPriority(const std::shared_ptr<ngraph::Node>& node) {
    if (node) {
        const auto& rtInfo = node->get_rt_info();
        auto it_info = rtInfo.find(PrimitivesPriority::get_type_info_static());
        if (it_info != rtInfo.end()) {
            return it_info->second.as<PrimitivesPriority>().value;
        } else {
            return {};
        }
    }
    return {};
}

Any PrimitivesPriority::merge(const ngraph::NodeVector& nodes) const {
    auto canBeMerged = [](const std::shared_ptr<Node>& node) -> bool {
        if (std::dynamic_pointer_cast<ngraph::opset1::Convolution>(node) ||
            std::dynamic_pointer_cast<ngraph::opset1::GroupConvolution>(node) ||
            std::dynamic_pointer_cast<ngraph::opset1::GroupConvolutionBackpropData>(node) ||
            std::dynamic_pointer_cast<ngraph::opset1::ConvolutionBackpropData>(node) ||
            std::dynamic_pointer_cast<ngraph::opset1::MatMul>(node)) {
            return true;
        }
        return false;
    };

    std::set<std::string> unique_pp;

    for (auto& node : nodes) {
        if (canBeMerged(node)) {
            std::string pp = getPrimitivesPriority(node);
            if (!pp.empty())
                unique_pp.insert(pp);
        }
    }

    if (unique_pp.size() > 1) {
        throw ngraph_error("PrimitivesPriority no rule defined for multiple values.");
    }

    std::string final_primitives_priority;
    if (unique_pp.size() == 1) {
        final_primitives_priority = *unique_pp.begin();
    }
    return PrimitivesPriority(final_primitives_priority);
}

bool PrimitivesPriority::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("value", value);
    return true;
}

std::string PrimitivesPriority::to_string() const {
    return value;
}
