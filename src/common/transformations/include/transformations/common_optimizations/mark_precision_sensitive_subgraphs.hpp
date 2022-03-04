// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MarkPrecisionSensitiveSubgraphs;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief MarkPrecisionSensitiveSubgraphs transformation marks the constants
 * inside the subgraph starting from precision-sensitive input and ending at
 * the ShapeOf node as disabled for FP16 compression.
 */
class ov::pass::MarkPrecisionSensitiveSubgraphs : public ModelPass {
public:
    OPENVINO_RTTI("MarkPrecisionSensitiveSubgraphs", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& f) override;
};
