// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <vector>

namespace ov {
namespace pass {

class NGRAPH_API SetBatchSize;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief Generic caller for all the transformations responsible to make model reshape-able by batch dimension
 */

class ov::pass::SetBatchSize : public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("SetBatchSize", "0");
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;
};

namespace ngraph {
namespace pass {
using ov::pass::SetBatchSize;
}  // namespace pass
}  // namespace ngraph
