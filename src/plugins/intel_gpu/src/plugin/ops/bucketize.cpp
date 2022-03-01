// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/bucketize.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/primitives/bucketize.hpp"

namespace ov {
namespace runtime {
namespace intel_gpu {

static void CreateBucketizeOp(Program& p, const std::shared_ptr<ngraph::op::v3::Bucketize>& op) {
    p.ValidateInputs(op, {2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);
    auto BucketizePrim = cldnn::bucketize(layerName,
                                          inputPrimitives[0],
                                          inputPrimitives[1],
                                          DataTypeFromPrecision(op->get_output_type()),
                                          op->get_with_right_bound(),
                                          op->get_friendly_name());

    p.AddPrimitive(BucketizePrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v3, Bucketize);

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
