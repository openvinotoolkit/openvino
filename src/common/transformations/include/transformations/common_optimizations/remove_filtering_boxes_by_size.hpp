// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API FuseFilteringBoxesBySize;
class TRANSFORMATIONS_API RemoveFilteringBoxesBySize;

}  // namespace pass
}  // namespace ov

class ov::pass::FuseFilteringBoxesBySize : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("FuseFilteringBoxesBySize", "0");
    FuseFilteringBoxesBySize();
};

class ov::pass::RemoveFilteringBoxesBySize : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RemoveFilteringBoxesBySize", "0");
    RemoveFilteringBoxesBySize();
};
