// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API RestoreTracedBatch;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Restores the leading Reshape target dimension that model conversion froze to one.
 *
 * Tracing a model with batch one turns expressions like `int(x.shape[0])` into a constant, so the traced Reshape keeps
 * the batch at one and a sibling `-1` silently absorbs the batch instead. The transformation does not infer which
 * dimension is the batch. It first propagates a symbol seeded on the leading input dimension and only acts when some
 * Reshape rebuilds that symbol out of a leading dimension of one, which means the batch no longer comes from the data.
 * It then reuses the `Gather(ShapeOf(parameter), 0)` expression the model already applies to other Reshape targets,
 * rewriting a target only when the leading element is a constant one, another element is `-1`, and the reshaped tensor
 * has a dynamic leading dimension. Models without such an expression are left unchanged.
 *
 * The transformation is not part of the SmartReshape pipeline and has to be registered explicitly.
 */
class ov::pass::RestoreTracedBatch : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("RestoreTracedBatch");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};
