// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

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
 * @brief This transformation is an entry point for OpenVINO transformations that
 * will be applied inside MOC. This transformations container is filled with
 * legacy transformations to reach parity between legacy front-ends and new
 * frontends calling from the Model Optimizer. It contains transformations to
 * avoid limitations of OpenVINO 1.X API such as unsupported INT64 for inputs,
 * usage of NCHW layout that is critical for TensorFlow models.
 */

class MOCLegacyTransformations : public ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("MOCLegacyTransformations");
    explicit MOCLegacyTransformations(const std::vector<std::string>& params_with_custom_types)
        : m_params_with_custom_types(params_with_custom_types) {}
    bool run_on_model(const std::shared_ptr<ov::Model>& f) override;

private:
    std::vector<std::string> m_params_with_custom_types;
};

}  // namespace pass
}  // namespace ov
