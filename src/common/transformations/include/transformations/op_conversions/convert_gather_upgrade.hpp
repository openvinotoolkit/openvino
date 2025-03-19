// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertGather1ToGather7;

class TRANSFORMATIONS_API ConvertGather7ToGather8;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ConvertGather1ToGather7 converts v1::Gather into v7::Gather.
 */
class ov::pass::ConvertGather1ToGather7 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertGather1ToGather7");
    ConvertGather1ToGather7();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief ConvertGather7ToGather8 converts v7::Gather into v8::Gather.
 */
class ov::pass::ConvertGather7ToGather8 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertGather7ToGather8");
    ConvertGather7ToGather8();
};
