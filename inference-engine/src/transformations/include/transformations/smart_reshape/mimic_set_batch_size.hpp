// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <numeric>

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MimicSetBatchSize;

}  // namespace pass
}  // namespace ov

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

class ov::pass::MimicSetBatchSize : public ov::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_function(std::shared_ptr<ov::Function> f) override;
};
