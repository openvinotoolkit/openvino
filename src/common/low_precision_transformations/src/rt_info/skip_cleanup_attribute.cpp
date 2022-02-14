// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/skip_cleanup_attribute.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <iterator>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include "low_precision/network_helper.hpp"
#include "low_precision/layer_transformation.hpp"

using namespace ngraph;
using namespace ov;

SkipCleanupAttribute::SkipCleanupAttribute(const bool skip)
    :
    SharedAttribute(skip) {
}

ov::Any SkipCleanupAttribute::create(
    const std::shared_ptr<ngraph::Node>& node,
    const bool skip) {
    auto& rt = node->get_rt_info();
    return (rt[SkipCleanupAttribute::get_type_info_static()] = SkipCleanupAttribute(skip));
}

std::string SkipCleanupAttribute::to_string() const {
    std::stringstream ss;
    ss << "SkipCleanup: {";
    attribute ? ss << "True" : ss << "False";
    ss << "}";
    return ss.str();
}
