// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertCompressedOnlyToLegacy;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief ConvertCompressedOnlyToLegacy transformation converts compression only FP16 format to legacy FP16 format.
 */
class ov::pass::ConvertCompressedOnlyToLegacy : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ConvertCompressedOnlyToLegacy");
    bool run_on_model(const std::shared_ptr<Model>& f) override;
};
