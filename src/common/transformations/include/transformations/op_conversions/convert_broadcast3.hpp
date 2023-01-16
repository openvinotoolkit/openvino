// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertBroadcast3;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertBroadcast3 : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertBroadcast3", "0");
    ConvertBroadcast3();
};
