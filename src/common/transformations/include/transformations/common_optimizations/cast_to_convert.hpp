// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API CastToConvert;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief CastToConvert transformation replaces single Cast operation with Convert operations with no_clamp=true and
 * use_rounding=true attributes.
 *
 * onnx's cast op is using no_clamp and rounding cast. when running single node cast test, will got different
 * result compared to ov's convert op, check
 * https://github.com/microsoft/onnxruntime/blob/bac0bff72b1b4e6fd68ae759a32644defac61944/onnxruntime/test/providers/cpu/tensor/cast_op_test.cc#L959
 * for example, float to int4, input value 31.9
 *   onnx cast:                               31.9 -> 32 -> 0x20 -> 0 (round and no_clamp)
 *   ov convert - default:                    31.9 -> 31 -> 7         (trunc and clamp)
 * so when running single node unit test with cast op, we use ov::op::v0::Convert with no_clamp=true and
 * use_rounding=true to align with onnx cast op behavior.
 * return {std::make_shared<v0::Convert>(data, elem_type, true, true)};
 */

class ov::pass::CastToConvert : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("CastToConvert");
    CastToConvert();
};
