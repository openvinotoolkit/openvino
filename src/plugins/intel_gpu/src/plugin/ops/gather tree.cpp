// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/gather_tree.hpp"

#include "intel_gpu/primitives/gather_tree.hpp"
#include "intel_gpu/primitives/reorder.hpp"

namespace ov {
namespace intel_gpu {

static void CreateGatherTreeOp(Program& p, const std::shared_ptr<ngraph::op::v1::GatherTree>& op) {
    p.ValidateInputs(op, {4});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    std::vector<cldnn::primitive_id> reorderedInputs;
    reorderedInputs.resize(inputPrimitives.size());

    for (size_t portIndex = 0; portIndex < inputPrimitives.size(); portIndex++) {
        auto inputDataType = DataTypeFromPrecision(op->get_input_element_type(portIndex));
        if (inputDataType == cldnn::data_types::i64) {
            // GPU primitive does not support i64 inputs,
            // so we need additional reorders to convert them to i32
            auto reorderPrimName = inputPrimitives[portIndex] + "_" + op->get_friendly_name() + Program::m_preProcessTag;
            auto targetFormat = DefaultFormatForDims(op->get_input_shape(portIndex).size());
            auto preprocessPrim = cldnn::reorder(reorderPrimName,
                                                 inputPrimitives[portIndex],
                                                 targetFormat,
                                                 cldnn::data_types::i32,
                                                 std::vector<float>(),
                                                 cldnn::reorder_mean_mode::subtract,
                                                 op->get_friendly_name());
            p.AddPrimitive(preprocessPrim);
            p.AddInnerPrimitiveToProfiler(reorderPrimName, layerName, op);
            reorderedInputs[portIndex] = reorderPrimName;
        } else {
            reorderedInputs[portIndex] = inputPrimitives[portIndex];
        }
    }

    auto gatherTreePrim = cldnn::gather_tree(layerName,
                                             reorderedInputs[0],
                                             reorderedInputs[1],
                                             reorderedInputs[2],
                                             reorderedInputs[3],
                                             op->get_friendly_name());

    p.AddPrimitive(gatherTreePrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v1, GatherTree);

}  // namespace intel_gpu
}  // namespace ov
