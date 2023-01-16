// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <utility>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertMulticlassNms8ToMulticlassNms9;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertMulticlassNms8ToMulticlassNms9 : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertMulticlassNms8ToMulticlassNms9", "0");
    ConvertMulticlassNms8ToMulticlassNms9();
};

namespace ngraph {
namespace pass {
using ov::pass::ConvertMulticlassNms8ToMulticlassNms9;
}  // namespace pass
}  // namespace ngraph
