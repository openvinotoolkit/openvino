// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <utility>
#include <memory>

#include <openvino/core/ov_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class OPENVINO_API ConvertNMSToNMSIEInternal;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertNMSToNMSIEInternal: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertNMSToNMSIEInternal();
};
