// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <utility>
#include <vector>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertNMS9ToNMSIEInternal;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertNMS9ToNMSIEInternal : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertNMS9ToNMSIEInternal", "0");
    ConvertNMS9ToNMSIEInternal();
};

namespace ngraph {
namespace pass {
using ov::pass::ConvertNMS9ToNMSIEInternal;
}  // namespace pass
}  // namespace ngraph
