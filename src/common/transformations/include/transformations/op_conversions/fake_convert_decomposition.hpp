// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API FakeConvertDecomposition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief FakeConvertDecomposition transformation decomposes FakeConvert layer.
 * f8: f8e4m3, f8e5m2
 * downconvert: f32->f8, f16->f8, bf16->f8
 * upconvert: f8->f32, f8->f16, f8->bf16
 * output = (upconvert(downconvert(input * scale - shift)) + shift) / scale
 *
 */

class ov::pass::FakeConvertDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FakeConvertDecomposition");
    FakeConvertDecomposition();
};
