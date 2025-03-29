// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/softmax.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/softmax.hpp"

namespace ov::intel_gpu {

static void CreateSoftmaxOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::Softmax>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);
    auto softmaxPrim = cldnn::softmax(layerName,
                                      inputs[0],
                                      op->get_axis());
    p.add_primitive(*op, softmaxPrim);
}

static void CreateSoftmaxOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v8::Softmax>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    int64_t axis = ov::util::try_normalize_axis(op->get_axis(), op->get_input_partial_shape(0).rank(), *op);

    auto softmaxPrim = cldnn::softmax(layerName,
                                      inputs[0],
                                      axis);
    p.add_primitive(*op, softmaxPrim);
}

static void CreateLogSoftmaxOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v5::LogSoftmax>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);
    std::string layerNameSoftmax = layer_type_name_ID(op) + "_softmax";

    int64_t axis = ov::util::try_normalize_axis(op->get_axis(), op->get_input_partial_shape(0).rank(), *op);

    auto softmaxPrim = cldnn::softmax(layerNameSoftmax,
                                      inputs[0],
                                      axis);

    auto logPrim = cldnn::activation(layerName, cldnn::input_info(layerNameSoftmax), cldnn::activation_func::log, {(0.0F), (0.0F)});

    p.add_primitive(*op, softmaxPrim);
    p.add_primitive(*op, logPrim);
}

REGISTER_FACTORY_IMPL(v1, Softmax);
REGISTER_FACTORY_IMPL(v8, Softmax);
REGISTER_FACTORY_IMPL(v5, LogSoftmax);

}  // namespace ov::intel_gpu
