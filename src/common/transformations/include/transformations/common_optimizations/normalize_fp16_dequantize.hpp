// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API NormalizeDequantizeFP16;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief NormalizeDequantizeFP16 handles the case where the Dequantize output
 * is FP16 instead of FP32. In ONNX QDQ models, Quantize is decomposed into a
 * FakeQuantize followed by a Convert(->int), and Dequantize is decomposed into
 * Convert(int->float) -> [Subtract(zp)] -> Multiply(scale). When the Dequantize
 * output is FP16, ConvertQuantizeDequantize cannot match the pattern and fuse the
 * entire sequence back into a single FakeQuantize.
 *
 * This pass rewrites the Dequantize output from FP16 to FP32 by inserting a
 * single FP16 cast at the end, leaving the FakeQuantize unchanged:
 *
 *   Convert(int->f16) -> [Subtract(f16_zp)] -> Multiply(f16_scale)
 *
 * becomes:
 *
 *   Convert(int->f32) -> [Subtract(f32_zp)] -> Multiply(f32_scale) -> Convert(f32->f16)
 *
 * ConvertQuantizeDequantize can then detect and fuse the full sequence into a
 * single FakeQuantize, eliminating the Convert, Subtract, and Multiply nodes.
 */
class ov::pass::NormalizeDequantizeFP16 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("NormalizeDequantizeFP16");
    NormalizeDequantizeFP16();
};
