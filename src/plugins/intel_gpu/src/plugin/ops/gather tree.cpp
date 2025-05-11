// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/gather_tree.hpp"

#include "intel_gpu/primitives/gather_tree.hpp"
#include "intel_gpu/primitives/reorder.hpp"

namespace ov::intel_gpu {

static void CreateGatherTreeOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v1::GatherTree>& op) {
    validate_inputs_count(op, {4});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    std::vector<cldnn::input_info> reordered_inputs;
    reordered_inputs.resize(inputs.size());

    for (size_t portIndex = 0; portIndex < inputs.size(); portIndex++) {
        auto inputDataType = cldnn::element_type_to_data_type(op->get_input_element_type(portIndex));
        if (inputDataType == cldnn::data_types::i64) {
            // GPU primitive does not support i64 inputs,
            // so we need additional reorders to convert them to i32
            auto reorderPrimName = inputs[portIndex].pid + "_" + op->get_friendly_name() + ProgramBuilder::m_preProcessTag;
            auto targetFormat = cldnn::format::get_default_format(op->get_input_shape(portIndex).size());
            auto preprocessPrim = cldnn::reorder(reorderPrimName,
                                                 inputs[portIndex],
                                                 targetFormat,
                                                 cldnn::data_types::i32);
            p.add_primitive(*op, preprocessPrim);
            reordered_inputs[portIndex] = cldnn::input_info(reorderPrimName);
        } else {
            reordered_inputs[portIndex] = inputs[portIndex];
        }
    }

    auto gatherTreePrim = cldnn::gather_tree(layerName,
                                             reordered_inputs[0],
                                             reordered_inputs[1],
                                             reordered_inputs[2],
                                             reordered_inputs[3]);

    p.add_primitive(*op, gatherTreePrim);
}

REGISTER_FACTORY_IMPL(v1, GatherTree);

}  // namespace ov::intel_gpu
