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

class TRANSFORMATIONS_API StridedSliceSqueeze;
class TRANSFORMATIONS_API SqueezeStridedSlice;
class TRANSFORMATIONS_API SharedSqueeze;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief StridedSliceSqueeze transformation looks for SS -> Squeeze and corrects SS inputs and attributes for SS output
 * to be squeeze-able
 */

class ov::pass::StridedSliceSqueeze : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("StridedSliceSqueeze", "0");
    StridedSliceSqueeze();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief StridedSliceSqueeze transformation looks for Squeeze -> SSe and corrects SS inputs and attributes for SS
 * output to be squeeze-able
 */

class ov::pass::SqueezeStridedSlice : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SqueezeStridedSlice", "0");
    SqueezeStridedSlice();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief SharedSqueeze transformation looks for shared Squeezes and leaves only one Squeeze reconnecting all the
 * outputs to it
 */

class ov::pass::SharedSqueeze : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("SharedSqueeze", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};
