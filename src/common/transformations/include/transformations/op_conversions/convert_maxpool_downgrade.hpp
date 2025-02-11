// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertMaxPool8ToMaxPool1;
class TRANSFORMATIONS_API ConvertMaxPool14ToMaxPool8;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ConvertMaxPool8ToMaxPool1 converts v8::MaxPool into v1::MaxPool.
 */
class ov::pass::ConvertMaxPool8ToMaxPool1 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertMaxPool8ToMaxPool1");
    ConvertMaxPool8ToMaxPool1();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief ConvertMaxPool14ToMaxPool8 converts v14::MaxPool into v8::MaxPool.
 */
class ov::pass::ConvertMaxPool14ToMaxPool8 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertMaxPool14ToMaxPool8");
    ConvertMaxPool14ToMaxPool8();
};
