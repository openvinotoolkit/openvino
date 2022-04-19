// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <numeric>

namespace ngraph {
namespace pass {

class NGRAPH_API MimicSetBatchSize;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief MimicSetBatchSize transformation relaxes hard-coded output batch dimension of Reshape operation.
 * For Reshape with input shape [in_batch, ...] and pattern value [out_batch, ...] it generates a sub-graph
 * which basically keeps ratio of input and output batch size and performs the following calculation:
 *
 * scale = float(out_batch) / float(in_batch)
 * modified_batch_dim = int(ceil(float(shape(input)[0]) * scale))
 *
 * This transformation should be executed only while setBatchSize method call
 */

class ngraph::pass::MimicSetBatchSize : public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("MimicSetBatchSize", "0");
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;
};
