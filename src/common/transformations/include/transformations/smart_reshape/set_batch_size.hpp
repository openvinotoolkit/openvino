// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API SetBatchSize;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief Generic caller for all the transformations responsible to make model reshape-able by batch dimension
 */

class ov::pass::SetBatchSize : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("SetBatchSize", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};
