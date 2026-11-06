// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/gguf/visibility.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov::frontend::gguf {
namespace pass {

// Moves a constant scale that follows a MatMul in front of it:  MatMul(A, W) * c  ->  MatMul(A * c, W)
// (exact, MatMul is linear in A). When A is itself a product -- the gated activation in front of an FFN
// down projection, silu(gate) * up -- the scale is applied to one factor, so neither the product nor the
// MatMul result is ever larger than the scaled value. This matters for models that scale a block's output
// by a constant below one (Granite's residual multiplier) while the block itself produces values beyond
// the f16 range on massive-activation tokens: the plugins execute the graph in f16, and the unscaled
// intermediate overflows to inf even though the scaled result is small.
// Only scales in [1/64, 1) are moved: larger ones would not reduce any magnitude, and much smaller ones
// would push the intermediate toward f16 subnormals.
//
// The frontend does not register this pass: the rewrite only helps a plugin that executes in f16, and on
// f32 it is pure overhead (the Multiply after a MatMul can fuse into it as a post-op, the one before cannot).
// Whether the target runs in f16 is a caller concern, so register it as a DecoderTransformationExtension:
//
//     ov::frontend::gguf::FrontEnd fe;
//     fe.add_extension(std::make_shared<ov::frontend::DecoderTransformationExtension>(
//         [](std::shared_ptr<ov::Model> model) {
//             ov::pass::Manager manager;
//             manager.register_pass<ov::frontend::gguf::pass::PushScaleBeforeMatMul>();
//             return manager.run_passes(model);
//         }));
//
// See also the "prescale_q" attribute of GGML_OP_FLASH_ATTN_EXT, the same rewrite for the attention scale.
class GGUF_FRONTEND_API PushScaleBeforeMatMul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::gguf::pass::PushScaleBeforeMatMul")
    PushScaleBeforeMatMul();
};

}  // namespace pass
}  // namespace ov::frontend::gguf
