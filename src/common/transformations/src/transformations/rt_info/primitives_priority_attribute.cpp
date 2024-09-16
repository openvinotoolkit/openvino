// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/primitives_priority_attribute.hpp"

#include <assert.h>

#include <functional>
#include <iterator>
#include <memory>
#include <ostream>

#include "openvino/core/node.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/matmul.hpp"

using namespace ov;

std::string ov::getPrimitivesPriority(const std::shared_ptr<ov::Node>& node) {
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

Any PrimitivesPriority::merge(const ov::NodeVector& nodes) const {
    auto canBeMerged = [](const std::shared_ptr<Node>& node) -> bool {
        if (ov::as_type_ptr<ov::op::v1::Convolution>(node) || ov::as_type_ptr<ov::op::v1::GroupConvolution>(node) ||
            ov::as_type_ptr<ov::op::v1::GroupConvolutionBackpropData>(node) ||
            ov::as_type_ptr<ov::op::v1::ConvolutionBackpropData>(node) || ov::as_type_ptr<ov::op::v0::MatMul>(node)) {
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
        OPENVINO_THROW("PrimitivesPriority no rule defined for multiple values.");
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
