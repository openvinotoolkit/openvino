// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/rt_info/skip_cleanup_attribute.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <iterator>
#include <vector>

using namespace ngraph;
using namespace ov;

ov::Any SkipCleanupAttribute::create(
    const std::shared_ptr<ngraph::Node>& node) {
    auto& rt = node->get_rt_info();
    return (rt[SkipCleanupAttribute::get_type_info_static()] = SkipCleanupAttribute());
}
