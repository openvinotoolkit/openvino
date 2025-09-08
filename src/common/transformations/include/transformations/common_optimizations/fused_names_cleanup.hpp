// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API FusedNamesCleanup;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief FusedNamesCleanup removes fused_names attribute
 */
class ov::pass::FusedNamesCleanup : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("FusedNamesCleanup");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};
