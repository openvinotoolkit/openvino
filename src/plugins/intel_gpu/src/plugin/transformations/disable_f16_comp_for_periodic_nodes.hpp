// Copyright (C) 2018-2025 Intel Corporation
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
#include "intel_gpu/runtime/execution_config.hpp"

namespace ov {
namespace intel_gpu {

/**
 * @ingroup ov_transformation_common_api
 * @brief DisableFP16CompressionForPeriodicFuncs transformation helps to set runtime info attributes in a single place.
 *
 * Every runtime info attribute that needs to be initialized should be registered
 * in run_on_function method. Also do not forget to override init methods for registered
 * attribute.
 * This transformations should be called first in transformation pipeline. If attribute was
 * already set initialization will be skipped for this node.
 */
class DisableFP16CompressionForPeriodicFuncs : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableFP16CompressionForPeriodicFuncs");
    DisableFP16CompressionForPeriodicFuncs();
};
}  // namespace intel_gpu
}  // namespace ov
