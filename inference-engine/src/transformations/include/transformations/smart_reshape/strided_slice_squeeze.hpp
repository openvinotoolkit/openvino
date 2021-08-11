// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API StridedSliceSqueeze;
class TRANSFORMATIONS_API SqueezeStridedSlice;
class TRANSFORMATIONS_API SharedSqueeze;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief StridedSliceSqueeze transformation looks for SS -> Squeeze and corrects SS inputs and attributes for SS output to be squeeze-able
 */

class ov::pass::StridedSliceSqueeze : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    StridedSliceSqueeze();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief StridedSliceSqueeze transformation looks for Squeeze -> SSe and corrects SS inputs and attributes for SS output to be squeeze-able
 */

class ov::pass::SqueezeStridedSlice : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SqueezeStridedSlice();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief SharedSqueeze transformation looks for shared Squeezes and leaves only one Squeeze reconnecting all the outputs to it
 */

class ov::pass::SharedSqueeze : public ov::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_function(std::shared_ptr<ov::Function> f) override;
};
