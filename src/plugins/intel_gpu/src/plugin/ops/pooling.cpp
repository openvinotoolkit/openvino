// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/avg_pool.hpp"

#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/pooling.hpp"

namespace ov {
namespace intel_gpu {

static void CreateAvgPoolOp(Program& p, const std::shared_ptr<ngraph::op::v1::AvgPool>& op) {
    p.ValidateInputs(op, {1});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto kernel = op->get_kernel();
    auto strides = op->get_strides();
    auto pads_begin = op->get_pads_begin();
    auto pads_end = op->get_pads_end();

    // Extend 1d vectors to 2d as 1d can't be handled properly by the graph optimizer for now
    kernel.resize(std::max<size_t>(2, kernel.size()), 1);
    strides.resize(std::max<size_t>(2, strides.size()), 1);
    pads_begin.resize(std::max<size_t>(2, pads_begin.size()), 0);
    pads_end.resize(std::max<size_t>(2, pads_end.size()), 0);

    auto poolPrim = cldnn::pooling(layerName,
                                   inputPrimitives[0],
                                   op->get_exclude_pad() ? cldnn::pooling_mode::average_no_padding : cldnn::pooling_mode::average,
                                   kernel,
                                   strides,
                                   pads_begin,
                                   tensor_from_dims(op->get_output_shape(0)),
                                   DataTypeFromPrecision(op->get_output_element_type(0)),
                                   op->get_friendly_name());
    poolPrim.pad_end = pads_end;
    p.AddPrimitive(poolPrim);
    p.AddPrimitiveToProfiler(op);
}

static void CreateMaxPoolOp(Program& p, const std::shared_ptr<ngraph::op::v1::MaxPool>& op) {
    p.ValidateInputs(op, {1});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto kernel = op->get_kernel();
    auto strides = op->get_strides();
    auto pads_begin = op->get_pads_begin();
    auto pads_end = op->get_pads_end();

    // Extend 1d vectors to 2d as 1d can't be handled properly by the graph optimizer for now
    kernel.resize(std::max<size_t>(2, kernel.size()), 1);
    strides.resize(std::max<size_t>(2, strides.size()), 1);
    pads_begin.resize(std::max<size_t>(2, pads_begin.size()), 0);
    pads_end.resize(std::max<size_t>(2, pads_end.size()), 0);

    auto poolPrim = cldnn::pooling(layerName,
                                   inputPrimitives[0],
                                   cldnn::pooling_mode::max,
                                   kernel,
                                   strides,
                                   pads_begin,
                                   tensor_from_dims(op->get_output_shape(0)),
                                   DataTypeFromPrecision(op->get_output_element_type(0)),
                                   op->get_friendly_name());
    poolPrim.pad_end = pads_end;
    p.AddPrimitive(poolPrim);
    p.AddPrimitiveToProfiler(op);
}

static void CreateMaxPoolOp(Program& p, const std::shared_ptr<ngraph::op::v8::MaxPool>& op) {
    p.ValidateInputs(op, {1});
    if (op->get_output_size() != 2) {
        IE_THROW() << "MaxPool opset 8 requires 2 outputs";
    }
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    const auto layer_type_name = layer_type_name_ID(op);
    const auto layerName = layer_type_name + ".0";

    const auto mutable_precision = op->get_output_element_type(1);
    const auto output_shape = op->get_output_shape(1);
    cldnn::layout mutableLayout = cldnn::layout(DataTypeFromPrecision(mutable_precision),
                                                DefaultFormatForDims(output_shape.size()),
                                                tensor_from_dims(output_shape));
    const auto shared_memory = p.GetEngine().allocate_memory(mutableLayout);
    const cldnn::primitive_id maxpool_mutable_id_w = layer_type_name + "_md_write";
    const auto op_friendly_name = op->get_friendly_name();
    const auto indices_mutable_prim = cldnn::mutable_data(maxpool_mutable_id_w,
                                                          shared_memory,
                                                          op_friendly_name);
    p.primitiveIDs[maxpool_mutable_id_w] = maxpool_mutable_id_w;
    p.AddPrimitive(indices_mutable_prim);
    inputPrimitives.push_back(maxpool_mutable_id_w);

    auto kernel = op->get_kernel();
    auto strides = op->get_strides();
    auto pads_begin = op->get_pads_begin();
    auto pads_end = op->get_pads_end();
    auto dilations = op->get_dilations();

    // Extend 1d vectors to 2d as 1d can't be handled properly by the graph optimizer for now
    kernel.resize(std::max<size_t>(2, kernel.size()), 1);
    strides.resize(std::max<size_t>(2, strides.size()), 1);
    pads_begin.resize(std::max<size_t>(2, pads_begin.size()), 0);
    pads_end.resize(std::max<size_t>(2, pads_end.size()), 0);
    dilations.resize(std::max<size_t>(2, dilations.size()), 1);

    auto poolPrim = cldnn::pooling(layerName,
                                   inputPrimitives[0],
                                   inputPrimitives.back(),
                                   kernel,
                                   strides,
                                   dilations,
                                   pads_begin,
                                   pads_end,
                                   op->get_axis(),
                                   DataTypeFromPrecision(op->get_index_element_type()),
                                   tensor_from_dims(op->get_output_shape(0)),
                                   DataTypeFromPrecision(op->get_output_element_type(0)),
                                   op_friendly_name);
    p.AddPrimitive(poolPrim);

    const cldnn::primitive_id maxpool_mutable_id_r = layer_type_name + ".1";
    const auto indices_mutable_id_r = cldnn::mutable_data(maxpool_mutable_id_r,
                                                          { layerName },
                                                          shared_memory,
                                                          op_friendly_name);
    p.primitiveIDs[maxpool_mutable_id_r] = maxpool_mutable_id_r;
    p.AddPrimitive(indices_mutable_id_r);

    p.AddPrimitiveToProfiler(poolPrim, op);
}


REGISTER_FACTORY_IMPL(v1, MaxPool);
REGISTER_FACTORY_IMPL(v8, MaxPool);
REGISTER_FACTORY_IMPL(v1, AvgPool);

}  // namespace intel_gpu
}  // namespace ov
