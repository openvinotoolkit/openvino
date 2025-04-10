// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/reorg_yolo.hpp"

#include "intel_gpu/primitives/reorg_yolo.hpp"

namespace ov::intel_gpu {

static void CreateReorgYoloOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v0::ReorgYolo>& op) {
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    uint32_t stride = static_cast<uint32_t>(op->get_strides()[0]);

    auto reorgPrim = cldnn::reorg_yolo(layerName,
                                       inputs[0],
                                       stride);

    p.add_primitive(*op, reorgPrim);
}

REGISTER_FACTORY_IMPL(v0, ReorgYolo);

}  // namespace ov::intel_gpu
