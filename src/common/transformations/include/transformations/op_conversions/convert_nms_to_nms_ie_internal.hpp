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

class TRANSFORMATIONS_API ConvertNMSToNMSIEInternal;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertNMSToNMSIEInternal : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertNMSToNMSIEInternal", "0");
    ConvertNMSToNMSIEInternal();
};

namespace ngraph {
namespace pass {
using ov::pass::ConvertNMSToNMSIEInternal;
}  // namespace pass
}  // namespace ngraph
