// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

/**
 * @brief Defines cleaning node runtime information pass
 * @file clean_rt_info.hpp
 */

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API CleanRtInfo;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief CleanRtInfo transformation helps to clean runtime info attributes.
 */
class ov::pass::CleanRtInfo : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("CleanRtInfo", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};
