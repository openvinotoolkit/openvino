// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations/transpose_sinking/ts_base.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
namespace transpose_sinking {

class TRANSFORMATIONS_API TSStridedSliceForward;
class TRANSFORMATIONS_API TSStridedSliceBackward;

}  // namespace transpose_sinking
}  // namespace pass
}  // namespace ov

class ov::pass::transpose_sinking::TSStridedSliceForward : public ov::pass::transpose_sinking::TSForwardBase {
public:
    OPENVINO_RTTI("ov::pass::TSStridedSliceForward", "0");
    TSStridedSliceForward();
};

class ov::pass::transpose_sinking::TSStridedSliceBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TSStridedSliceBackward", "0");
    TSStridedSliceBackward();
};
