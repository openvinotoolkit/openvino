// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/softmax.hpp"
#include "ngraph/op/log_softmax.hpp"

#include "api/softmax.hpp"
#include "api/activation.hpp"

namespace CLDNNPlugin {

static cldnn::softmax::dimension_t GetSoftmaxAxis(int64_t axis, size_t rank) {
    switch (axis) {
    // FIXME: it seems that axis=0 should correspond to normalize_b;
    case 0: return cldnn::softmax::normalize_all;
    case 1: return cldnn::softmax::normalize_f;
    case 2:
        if (rank > 4)
            return cldnn::softmax::normalize_z;
        else
            return cldnn::softmax::normalize_y;
    case 3:
        if (rank > 4)
            return cldnn::softmax::normalize_y;
        else
            return cldnn::softmax::normalize_x;
    case 4:
        return cldnn::softmax::normalize_x;
    default: THROW_IE_EXCEPTION << "Invalid softmax axis " << axis;
    }
    return cldnn::softmax::normalize_fyx;
}

void CreateSoftmaxOp(Program& p, const std::shared_ptr<ngraph::op::v1::Softmax>& op) {
    p.ValidateInputs(op, {1});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);
    auto softmaxPrim = cldnn::softmax(layerName,
                                      inputPrimitives[0],
                                      GetSoftmaxAxis(op->get_axis(), op->get_input_shape(0).size()));
    p.AddPrimitive(softmaxPrim);
    p.AddPrimitiveToProfiler(op);
}

void CreateLogSoftmaxOp(Program& p, const std::shared_ptr<ngraph::op::v5::LogSoftmax>& op) {
    p.ValidateInputs(op, {1});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);
    std::string layerNameSoftmax = layer_type_name_ID(op) + "_softmax";

    auto axis = op->get_axis();
    if (axis < 0)
        axis += op->get_input_shape(0).size();

    auto softmaxPrim = cldnn::softmax(layerNameSoftmax,
                                      inputPrimitives[0],
                                      GetSoftmaxAxis(static_cast<size_t>(axis), op->get_input_shape(0).size()));

    auto logPrim = cldnn::activation(layerName, layerNameSoftmax, cldnn::activation_func::log);

    p.AddPrimitive(softmaxPrim);
    p.AddPrimitive(logPrim);
    p.AddPrimitiveToProfiler(layerNameSoftmax, op);
    p.AddPrimitiveToProfiler(layerName, op);
}

REGISTER_FACTORY_IMPL(v1, Softmax);
REGISTER_FACTORY_IMPL(v5, LogSoftmax);

}  // namespace CLDNNPlugin
