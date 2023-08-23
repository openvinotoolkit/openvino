// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <numeric>

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

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

class ov::pass::MimicSetBatchSize : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("MimicSetBatchSize", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};
