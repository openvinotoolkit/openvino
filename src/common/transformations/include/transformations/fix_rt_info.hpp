// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

/**
 * @brief Defines initialize node runtime information pass
 * @file init_node_info.hpp
 */

#include <vector>
#include <memory>

#include <ngraph/pass/graph_rewrite.hpp>

/**
 * @brief ngraph namespace
 */
namespace ngraph {

/**
 * @brief ngraph::pass namespace
 */
namespace pass {

class NGRAPH_API FixRtInfo;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief FixRtInfo transformation helps to fix info attributes in a single place.
 *        User can pass runtime attribute using various types.
 *        This Pass should generalize them runtime info representation.
 *
 * Used to extract runtime attributes from shared pointer to `ov::RuntimeAttributeWrapper` to standard or trivial types
 */
class ngraph::pass::FixRtInfo: public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;
};
