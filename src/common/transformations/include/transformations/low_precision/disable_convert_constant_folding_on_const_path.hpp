// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include <openvino/core/ov_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class OPENVINO_API DisableConvertConstantFoldingOnConstPath;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::DisableConvertConstantFoldingOnConstPath : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    DisableConvertConstantFoldingOnConstPath(
        const element::TypeVector & inputPrecisions = {});
};
