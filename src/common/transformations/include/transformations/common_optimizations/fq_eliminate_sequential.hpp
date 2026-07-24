// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API FakeQuantizeEliminateSequential;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief FakeQuantizeEliminateSequential removes a redundant second FakeQuantize from a sequential
 * pair (FQ1 -> FQ2). The notation below is FQ(in_low, in_high, out_low, out_high, levels).
 *
 * FQ1 and FQ2 may be separated by one or several value-preserving ops.
 * A matched FakeQuantize is applied element-wise (whether per-tensor or per-channel), so it
 * commutes with such ops and the folding stays valid through the chain.
 *
 * FQ2 is removed (FQ1 kept) when it is identical to FQ1, i.e. it has the same levels, auto broadcast,
 * and range constants (input_low/input_high/output_low/output_high). In that case FQ2 only re-applies
 * the quantization already produced by FQ1 and is redundant.
 *
 * Before:
 *      ┌─────────┐
 *      │  Data   │
 *      └────┬────┘
 *           │
 *      ┌────┴────┐
 *      │   FQ1   │
 *      └────┬────┘
 *           │
 *    ┌──────┴───────┐
 *    │ Reshape /    │  (optional, zero or more value-preserving ops)
 *    │ Transpose /  │
 *    │ Squeeze /    │
 *    │ Unsqueeze    │
 *    └──────┬───────┘
 *           │
 *      ┌────┴────┐
 *      │   FQ2   │  (identical params to FQ1)
 *      └────┬────┘
 *           │
 *      ┌────┴────┐
 *      │Consumer │
 *      └─────────┘
 *
 * After:
 *      ┌─────────┐
 *      │  Data   │
 *      └────┬────┘
 *           │
 *      ┌────┴────┐
 *      │   FQ1   │
 *      └────┬────┘
 *           │
 *    ┌──────┴───────┐
 *    │ Reshape /    │  (optional, preserved)
 *    │ Transpose /  │
 *    │ Squeeze /    │
 *    │ Unsqueeze    │
 *    └──────┬───────┘
 *           │
 *      ┌────┴────┐
 *      │Consumer │
 *      └─────────┘
 *
 * For example: FQ1(-1, 1, -1, 1, 256) -> FQ2(-1, 1, -1, 1, 256)  =>  FQ1(-1, 1, -1, 1, 256)
 */
class ov::pass::FakeQuantizeEliminateSequential : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("FakeQuantizeEliminateSequential");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};
