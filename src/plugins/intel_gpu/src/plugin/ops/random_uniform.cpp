// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "ngraph/op/random_uniform.hpp"
#include "intel_gpu/primitives/random_uniform.hpp"


namespace ov {
namespace intel_gpu {

namespace {

void CreateRandomUniformOp(Program &p, const std::shared_ptr<ngraph::op::v8::RandomUniform> &op) {
    auto inputs = p.GetInputInfo(op);
    auto output_shape = op->get_output_shape(0);
    cldnn::format outputFormat = cldnn::format::get_default_format(output_shape.size());

    auto random_uniform_prim = cldnn::random_uniform(layer_type_name_ID(op),
                                                     inputs,
                                                     cldnn::element_type_to_data_type(op->get_out_type()),
                                                     op->get_global_seed(),
                                                     op->get_op_seed(),
                                                     tensor_from_dims(output_shape),
                                                     outputFormat);
    p.add_primitive(*op, random_uniform_prim);
}

} // namespace

REGISTER_FACTORY_IMPL(v8, RandomUniform);

}  // namespace intel_gpu
}  // namespace ov
