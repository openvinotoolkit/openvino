// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertGather7ToGather1;
class TRANSFORMATIONS_API ConvertGather8ToGather7;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ConvertGather7ToGather1 converts v7::Gather into v1::Gather.
 */
class ov::pass::ConvertGather7ToGather1 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertGather7ToGather1");
    ConvertGather7ToGather1();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief ConvertGather8ToGather7 converts v8::Gather into v7::Gather.
 */
class ov::pass::ConvertGather8ToGather7 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertGather8ToGather7");
    ConvertGather8ToGather7();
};
