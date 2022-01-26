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
namespace runtime {
namespace intel_gpu {

struct PoolingParameters {
    cldnn::tensor kernel;
    cldnn::tensor stride;
    cldnn::tensor dilation;
    cldnn::tensor pad_begin;
    cldnn::tensor pad_end;
};

static PoolingParameters GetPoolingParameters(const ngraph::Shape& kernel,
                                              const ngraph::Strides& strides,
                                              const ngraph::Shape& pads_begin,
                                              const ngraph::Shape& pads_end,
                                              const ngraph::Strides& dilations = {}) {
    cldnn::tensor k, s, pb, pe;
    cldnn::tensor d{cldnn::batch(1), cldnn::feature(1), cldnn::spatial(1, 1, 1)};
    const auto is_dilation_specified = !dilations.empty();

    if (pads_begin.size() != strides.size() || pads_end.size() != strides.size() || kernel.size() != strides.size()
        || (is_dilation_specified && dilations.size() != strides.size()))
        IE_THROW() << "Strides, KernelSizes, Pads (and Dilations, if specified) are supposed to have the same elements count";

    std::vector<cldnn::tensor::value_type> pb_casted(pads_begin.begin(), pads_begin.end());
    std::vector<cldnn::tensor::value_type> pe_casted(pads_end.begin(), pads_end.end());
    switch (strides.size()) {
        case 3: {
            k = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(kernel[2], kernel[1], kernel[0]));
            s = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(strides[2], strides[1], strides[0]));
            pb = cldnn::tensor(cldnn::batch(0), cldnn::feature(0), cldnn::spatial(pb_casted[2], pb_casted[1], pb_casted[0]));
            pe = cldnn::tensor(cldnn::batch(0), cldnn::feature(0), cldnn::spatial(pe_casted[2], pe_casted[1], pe_casted[0]));
            if (is_dilation_specified) {
                d = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(dilations[2], dilations[1], dilations[0]));
            }
            break;
        }
        case 2: {
            k = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(kernel[1], kernel[0], 1));
            s = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(strides[1], strides[0], 1));
            pb = cldnn::tensor(cldnn::batch(0), cldnn::feature(0), cldnn::spatial(pb_casted[1], pb_casted[0], 0));
            pe = cldnn::tensor(cldnn::batch(0), cldnn::feature(0), cldnn::spatial(pe_casted[1], pe_casted[0], 0));
            if (is_dilation_specified) {
                d = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(dilations[1], dilations[0], 1));
            }
            break;
        }
        case 1: {
            k = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(kernel[0], 1, 1));
            s = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(strides[0], 1, 1));
            pb = cldnn::tensor(cldnn::batch(0), cldnn::feature(0), cldnn::spatial(pb_casted[0], 0, 0));
            pe = cldnn::tensor(cldnn::batch(0), cldnn::feature(0), cldnn::spatial(pe_casted[0], 0, 0));
            if (is_dilation_specified) {
                d = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(dilations[0], 1, 1));
            }
            break;
        }
        default: IE_THROW() << "Unsupported pooling parameters size. Only 1d, 2d, and 3d cases are supported";
    }

    return {k, s, d, pb, pe};
}

static void CreateAvgPoolOp(Program& p, const std::shared_ptr<ngraph::op::v1::AvgPool>& op) {
    p.ValidateInputs(op, {1});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto params = GetPoolingParameters(op->get_kernel(), op->get_strides(), op->get_pads_begin(), op->get_pads_end());
    auto poolPrim = cldnn::pooling(layerName,
                                   inputPrimitives[0],
                                   op->get_exclude_pad() ? cldnn::pooling_mode::average_no_padding : cldnn::pooling_mode::average,
                                   params.kernel,
                                   params.stride,
                                   params.pad_begin,
                                   tensor_from_dims(op->get_output_shape(0)),
                                   DataTypeFromPrecision(op->get_output_element_type(0)),
                                   op->get_friendly_name());
    poolPrim.pad_end = params.pad_end;
    p.AddPrimitive(poolPrim);
    p.AddPrimitiveToProfiler(op);
}

static void CreateMaxPoolOp(Program& p, const std::shared_ptr<ngraph::op::v1::MaxPool>& op) {
    p.ValidateInputs(op, {1});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto params = GetPoolingParameters(op->get_kernel(), op->get_strides(), op->get_pads_begin(), op->get_pads_end());
    auto poolPrim = cldnn::pooling(layerName,
                                   inputPrimitives[0],
                                   cldnn::pooling_mode::max,
                                   params.kernel,
                                   params.stride,
                                   params.pad_begin,
                                   tensor_from_dims(op->get_output_shape(0)),
                                   DataTypeFromPrecision(op->get_output_element_type(0)),
                                   op->get_friendly_name());
    poolPrim.pad_end = params.pad_end;
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

    const auto params = GetPoolingParameters(op->get_kernel(), op->get_strides(), op->get_pads_begin(), op->get_pads_end(), op->get_dilations());
    auto poolPrim = cldnn::pooling(layerName,
                                   inputPrimitives[0],
                                   inputPrimitives.back(),
                                   params.kernel,
                                   params.stride,
                                   params.dilation,
                                   params.pad_begin,
                                   params.pad_end,
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
}  // namespace runtime
}  // namespace ov
