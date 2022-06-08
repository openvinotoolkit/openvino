// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/bucketize.hpp"

#include <ngraph/op/bucketize.hpp>

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program.hpp"

namespace ov {
namespace runtime {
namespace intel_gpu {

namespace {

void CreateBucketizeOp(Program& p, const std::shared_ptr<ngraph::op::v3::Bucketize>& op) {
    p.ValidateInputs(op, {2});

    const cldnn::bucketize bucketize_prim(layer_type_name_ID(op),
                                          p.GetInputPrimitiveIDs(op),
                                          DataTypeFromPrecision(op->get_output_type()),
                                          op->get_with_right_bound(),
                                          op->get_friendly_name());
    p.AddPrimitive(bucketize_prim);
    p.AddPrimitiveToProfiler(op);
}

}  // namespace

REGISTER_FACTORY_IMPL(v3, Bucketize);

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
