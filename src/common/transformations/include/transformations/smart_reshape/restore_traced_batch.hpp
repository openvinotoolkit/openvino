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
 * @brief Restores a Reshape target leading dimension that model tracing froze to one.
 *
 * Tracing with batch one turns expressions like `int(x.shape[0])` into constants, so a Reshape keeps a leading one
 * while a `-1` absorbs the batch. When symbolic shape inference proves the batch is rebuilt later, the constant is
 * replaced with the `Gather(ShapeOf(parameter), 0)` expression the model already uses for other Reshape targets.
 */
class ov::pass::RestoreTracedBatch : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("RestoreTracedBatch");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};
