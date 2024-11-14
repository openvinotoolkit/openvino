// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations/transpose_sinking/ts_base.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {
namespace transpose_sinking {

class TRANSFORMATIONS_API TSSliceForward;
class TRANSFORMATIONS_API TSSliceBackward;

}  // namespace transpose_sinking
}  // namespace pass
}  // namespace ov

class ov::pass::transpose_sinking::TSSliceForward : public ov::pass::transpose_sinking::TSForwardBase {
public:
    OPENVINO_RTTI("ov::pass::TSSliceForward", "0");
    TSSliceForward();
};

class ov::pass::transpose_sinking::TSSliceBackward : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::pass::TSSliceBackward", "0");
    TSSliceBackward();
};
