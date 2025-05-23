// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bucketize.hpp"
#include "intel_gpu/primitives/bucketize.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"

namespace ov::intel_gpu {

namespace {

void CreateBucketizeOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v3::Bucketize>& op) {
    validate_inputs_count(op, {2});

    const cldnn::bucketize bucketize_prim(layer_type_name_ID(op),
                                          p.GetInputInfo(op),
                                          cldnn::element_type_to_data_type(op->get_output_type()),
                                          op->get_with_right_bound());
    p.add_primitive(*op, bucketize_prim);
}

}  // namespace

REGISTER_FACTORY_IMPL(v3, Bucketize);

}  // namespace ov::intel_gpu
