// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/col2im.hpp"

#include "intel_gpu/primitives/col2im.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace intel_gpu {

static void CreateCol2ImOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v15::Col2Im>& op) {
    validate_inputs_count(op, {3});
    auto inputPrimitives = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    // The total number of blocks calculated(L) = product from d=1 to 2 of
    //  floor((output_size[d] + pads_begin[d] + pads_end[d] - dilation[d] * (kernel_size[d] - 1) - 1) / stride[d] + 1)
    //  d : all spatial dimension
    auto strides = op->get_strides();
    auto dilations = op->get_dilations();
    auto pads_begin = op->get_pads_begin();
    auto pads_end = op->get_pads_end();
    ov::CoordinateDiff padding_begin;
    ov::CoordinateDiff padding_end;

    for (auto p : op->get_pads_begin())
        padding_begin.push_back(p);
    for (auto p : op->get_pads_end())
        padding_end.push_back(p);

    auto output_shape_const = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    auto vec_output_shape = output_shape_const->cast_vector<size_t>();
    ov::Shape output_shape(vec_output_shape);

    auto kernel_size_const = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(2));
    auto kernel_size = kernel_size_const->cast_vector<size_t>();
    ov::Shape kernel_shape(kernel_size);

    // Create col2im prim
    // iputs : data, output size,  kernel_size(required)
    auto Col2ImPrim = cldnn::col2im(layerName,
                                            inputPrimitives[0],
                                            strides,
                                            dilations,
                                            padding_begin,
                                            padding_end,
                                            vec_output_shape,
                                            kernel_shape);

    p.add_primitive(*op, Col2ImPrim);
}

REGISTER_FACTORY_IMPL(v15, Col2Im);

}  // namespace intel_gpu
}  // namespace ov
