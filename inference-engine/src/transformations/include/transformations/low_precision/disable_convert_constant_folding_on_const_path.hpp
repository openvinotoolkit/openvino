// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API DisableConvertConstantFoldingOnConstPath;

}  // namespace pass
}  // namespace ov

class ov::pass::DisableConvertConstantFoldingOnConstPath : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    DisableConvertConstantFoldingOnConstPath(
        const element::TypeVector & inputPrecisions = {});
};
