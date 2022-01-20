// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/fused_names_attribute.hpp"

#include <assert.h>

#include <functional>
#include <iterator>
#include <memory>
#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include <ostream>

using namespace ngraph;
using namespace ov;

std::string FusedNames::getNames() const {
    std::string res;
    for (auto& name : fused_names) {
        res += (res.empty() ? name : "," + name);
    }
    return res;
}

std::vector<std::string> FusedNames::getVectorNames() const {
    return std::vector<std::string>(fused_names.begin(), fused_names.end());
}

void FusedNames::fuseWith(const FusedNames& names) {
    for (const auto& name : names.fused_names) {
        fused_names.insert(name);
    }
}

std::string ngraph::getFusedNames(const std::shared_ptr<ngraph::Node>& node) {
    if (node) {
        const auto& rtInfo = node->get_rt_info();
        auto it_info = rtInfo.find(FusedNames::get_type_info_static());
        if (it_info != rtInfo.end()) {
            return it_info->second.as<FusedNames>().getNames();
        } else {
            return {};
        }
    }
    return {};
}

std::vector<std::string> ngraph::getFusedNamesVector(const std::shared_ptr<ngraph::Node>& node) {
    if (node) {
        const auto& rtInfo = node->get_rt_info();
        auto it_info = rtInfo.find(FusedNames::get_type_info_static());
        if (it_info != rtInfo.end()) {
            return it_info->second.as<FusedNames>().getVectorNames();
        } else {
            return {};
        }
    }
    return {};
}

Any FusedNames::merge(const ngraph::NodeVector& nodes) const {
    FusedNames mergedNames;
    for (auto& node : nodes) {
        const auto& rtInfo = node->get_rt_info();
        auto it_info = rtInfo.find(FusedNames::get_type_info_static());
        if (it_info != rtInfo.end()) {
            mergedNames.fuseWith(it_info->second.as<FusedNames>());
        }
    }
    return mergedNames;
}

Any FusedNames::init(const std::shared_ptr<ngraph::Node>& node) const {
    return FusedNames{node->get_friendly_name()};
}

bool FusedNames::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("value", fused_names);
    return true;
}

std::string FusedNames::to_string() const {
    return getNames();
}
