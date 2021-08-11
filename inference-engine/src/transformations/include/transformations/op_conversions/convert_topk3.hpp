// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertTopK3;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertTopK3: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertTopK3();
};
