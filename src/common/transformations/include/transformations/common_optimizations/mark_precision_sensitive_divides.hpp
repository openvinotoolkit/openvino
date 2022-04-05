// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MarkPrecisionSensitiveDivides;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief MarkPrecisionSensitiveDivides transformation marks the Divide fp16 layers
 * inside the subgraph starting from precision-sensitive input and ending at
 * the ShapeOf node as disabled for ConvertDivide transformation.
 */
class ov::pass::MarkPrecisionSensitiveDivides : public ModelPass {
public:
    OPENVINO_RTTI("MarkPrecisionSensitiveDivides", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};
