// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MOCLegacyTransformations;

}  // namespace pass
}  // namespace ov

namespace ov {
namespace pass {

/**
 * @brief This transformation is an entry point for nGraph transformations that
 * will be applied inside MOC. This transformations container is filled with
 * legacy transformations to reach parity between legacy front-ends and new
 * frontends calling from the Model Optimizer. It contains transformations to
 * avoid limitations of OpenVINO 1.X API such as unsupported INT64 for inputs,
 * usage of NCHW layout that is critical for TensorFlow models.
 */

class MOCLegacyTransformations : public FunctionPass {
 public:
  OPENVINO_RTTI("ov::pass::MOCLegacyTransformations");
  bool run_on_function(std::shared_ptr<ov::Function> f) override;
};

}  // namespace pass
}  // namespace ov
