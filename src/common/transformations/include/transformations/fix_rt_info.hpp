// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

/**
 * @brief Defines initialize node runtime information pass
 * @file init_node_info.hpp
 */

#include <memory>
#include <openvino/pass/graph_rewrite.hpp>
#include <vector>

#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API FixRtInfo;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief FixRtInfo transformation helps to fix info attributes in a single place.
 *        User can pass runtime attribute using various types.
 *        This Pass should generalize them runtime info representation.
 *
 * Used to extract runtime attributes from shared pointer to `ov::RuntimeAttributeWrapper` to standard or trivial types
 */
class ov::pass::FixRtInfo : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("FixRtInfo", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};
