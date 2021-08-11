// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <utility>
#include <memory>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertMulticlassNmsToMulticlassNmsIE;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertMulticlassNmsToMulticlassNmsIE: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertMulticlassNmsToMulticlassNmsIE();
};
