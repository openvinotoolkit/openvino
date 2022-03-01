// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <utility>
#include <memory>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertNMSToNMSIEInternal;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertNMSToNMSIEInternal: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertNMSToNMSIEInternal", "0");
    ConvertNMSToNMSIEInternal();
};
