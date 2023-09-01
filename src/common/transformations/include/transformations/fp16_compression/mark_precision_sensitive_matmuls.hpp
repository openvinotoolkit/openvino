// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MarkPrecisionSensitiveMatmuls;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief MarkPrecisionSensitiveMatmuls adds Converts to keep mixed FP16/FP32 graph type consistent
 */
class ov::pass::MarkPrecisionSensitiveMatmuls : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("MarkPrecisionSensitiveMatmuls", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};
