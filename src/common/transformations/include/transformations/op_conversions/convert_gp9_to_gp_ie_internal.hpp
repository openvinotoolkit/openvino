// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertGP9ToGPIEInternal;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertGP9ToGPIEInternal : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertGP9ToGPIEInternal", "0");
    ConvertGP9ToGPIEInternal();
};
