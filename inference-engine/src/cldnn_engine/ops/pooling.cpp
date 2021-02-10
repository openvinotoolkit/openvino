// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/avg_pool.hpp"

#include "api/pooling.hpp"

namespace CLDNNPlugin {

struct PoolingParameters {
    cldnn::tensor kernel;
    cldnn::tensor stride;
    cldnn::tensor pad_begin;
    cldnn::tensor pad_end;
};

static PoolingParameters GetPoolingParameters(const ngraph::Shape& kernel,
                                              const ngraph::Strides& strides,
                                              const ngraph::Shape& pads_begin,
                                              const ngraph::Shape& pads_end) {
    cldnn::tensor k, s, pb, pe;
    if (pads_begin.size() != strides.size() || pads_end.size() != strides.size() || kernel.size() != strides.size())
        THROW_IE_EXCEPTION << "Strides, KernelSizes and Pads are supposed to have the same elements count";

    std::vector<cldnn::tensor::value_type> pb_casted(pads_begin.begin(), pads_begin.end());
    std::vector<cldnn::tensor::value_type> pe_casted(pads_end.begin(), pads_end.end());
    switch (strides.size()) {
        case 3: {
            k = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(kernel[2], kernel[1], kernel[0]));
            s = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(strides[2], strides[1], strides[0]));
            pb = cldnn::tensor(cldnn::batch(0), cldnn::feature(0), cldnn::spatial(-pb_casted[2], -pb_casted[1], -pb_casted[0]));
            pe = cldnn::tensor(cldnn::batch(0), cldnn::feature(0), cldnn::spatial(-pe_casted[2], -pe_casted[1], -pe_casted[0]));
            break;
        }
        case 2: {
            k = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(kernel[1], kernel[0], 1));
            s = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(strides[1], strides[0], 1));
            pb = cldnn::tensor(cldnn::batch(0), cldnn::feature(0), cldnn::spatial(-pb_casted[1], -pb_casted[0], 0));
            pe = cldnn::tensor(cldnn::batch(0), cldnn::feature(0), cldnn::spatial(-pe_casted[1], -pe_casted[0], 0));
            break;
        }
        case 1: {
            k = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(kernel[0], 1, 1));
            s = cldnn::tensor(cldnn::batch(1), cldnn::feature(1), cldnn::spatial(strides[0], 1, 1));
            pb = cldnn::tensor(cldnn::batch(0), cldnn::feature(0), cldnn::spatial(-pb_casted[0], 0, 0));
            pe = cldnn::tensor(cldnn::batch(0), cldnn::feature(0), cldnn::spatial(-pe_casted[0], 0, 0));
            break;
        }
        default: THROW_IE_EXCEPTION << "Unsupported pooling parameters size. Only 1d, 2d, and 3d cases are supported";
    }

    return {k, s, pb, pe};
}

void CreateAvgPoolOp(Program& p, const std::shared_ptr<ngraph::op::v1::AvgPool>& op) {
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
                                   CldnnTensorFromIEDims(op->get_output_shape(0)),
                                   DataTypeFromPrecision(op->get_output_element_type(0)));
    poolPrim.pad_end = params.pad_end;
    p.AddPrimitive(poolPrim);
    p.AddPrimitiveToProfiler(op);
}

void CreateMaxPoolOp(Program& p, const std::shared_ptr<ngraph::op::v1::MaxPool>& op) {
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
                                   CldnnTensorFromIEDims(op->get_output_shape(0)),
                                   DataTypeFromPrecision(op->get_output_element_type(0)));
    poolPrim.pad_end = params.pad_end;
    p.AddPrimitive(poolPrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v1, MaxPool);
REGISTER_FACTORY_IMPL(v1, AvgPool);

}  // namespace CLDNNPlugin
