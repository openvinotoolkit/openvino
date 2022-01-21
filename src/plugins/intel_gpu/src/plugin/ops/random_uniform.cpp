// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "ngraph/op/random_uniform.hpp"
#include "intel_gpu/primitives/random_uniform.hpp"


namespace ov {
namespace runtime {
namespace intel_gpu {

namespace {

void CreateRandomUniformOp(Program &p, const std::shared_ptr<ngraph::op::v8::RandomUniform> &op) {
    auto input_primitives = p.GetInputPrimitiveIDs(op);
    auto output_shape = op->get_output_shape(0);
    cldnn::format outputFormat = DefaultFormatForDims(output_shape.size());

    auto random_uniform_prim = cldnn::random_uniform(layer_type_name_ID(op),
                                                     input_primitives,
                                                     DataTypeFromPrecision(op->get_out_type()),
                                                     op->get_global_seed(),
                                                     op->get_op_seed(),
                                                     tensor_from_dims(output_shape),
                                                     outputFormat);
    p.AddPrimitive(random_uniform_prim);
    p.AddPrimitiveToProfiler(op);
}

} // namespace

REGISTER_FACTORY_IMPL(v8, RandomUniform);

} // namespace intel_gpu
} // namespace runtime
} // namespace ov
