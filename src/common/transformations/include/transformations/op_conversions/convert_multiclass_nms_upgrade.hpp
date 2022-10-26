// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <utility>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertMulticlassNms8ToMulticlassNms9;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertMulticlassNms8ToMulticlassNms9 : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertMulticlassNms8ToMulticlassNms9", "0");
    ConvertMulticlassNms8ToMulticlassNms9();
};
