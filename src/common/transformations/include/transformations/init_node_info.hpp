// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

/**
 * @brief Defines initialize node runtime information pass
 * @file init_node_info.hpp
 */

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API InitNodeInfo;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief InitNodeInfo transformation helps to set runtime info attributes in a single place.
 *
 * Every runtime info attribute that needs to be initialized should be registered
 * in run_on_function method. Also do not forget to override init methods for registered
 * attribute.
 * This transformations should be called first in transformation pipeline. If attribute was
 * already set initialization will be skipped for this node.
 */
class ov::pass::InitNodeInfo : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("InitNodeInfo", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};
