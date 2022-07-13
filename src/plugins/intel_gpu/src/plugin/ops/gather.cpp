// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/gather.hpp"

#include "intel_gpu/primitives/gather.hpp"
#include "intel_gpu/primitives/reorder.hpp"

namespace ov {
namespace intel_gpu {

template <typename T>
void CreateGatherOpBase(Program& p, const std::shared_ptr<T>& op, const int64_t batch_dim = 0, bool support_neg_ind = false) {
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    int64_t axis = op->get_axis();

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

    auto gatherPrim = cldnn::gather(layerName,
                                    reorderedInputs[0],
                                    reorderedInputs[1],
                                    axis,
                                    op->get_output_shape(0),
                                    batch_dim,
                                    support_neg_ind,
                                    op->get_friendly_name());

    p.AddPrimitive(gatherPrim);
    p.AddPrimitiveToProfiler(op);
}

static void CreateGatherOp(Program& p, const std::shared_ptr<ngraph::op::v1::Gather>& op) {
    p.ValidateInputs(op, {2, 3});
    CreateGatherOpBase<ngraph::op::v1::Gather>(p, op);
}

REGISTER_FACTORY_IMPL(v1, Gather);

static void CreateGatherOp(Program& p, const std::shared_ptr<ngraph::op::v7::Gather>& op) {
    p.ValidateInputs(op, {2, 3, 4});
    CreateGatherOpBase<ngraph::op::v7::Gather>(p, op, op->get_batch_dims());
}

REGISTER_FACTORY_IMPL(v7, Gather);

static void CreateGatherOp(Program& p, const std::shared_ptr<ngraph::op::v8::Gather>& op) {
    p.ValidateInputs(op, {2, 3, 4});
    CreateGatherOpBase<ngraph::op::v8::Gather>(p, op, op->get_batch_dims(), true);
}

REGISTER_FACTORY_IMPL(v8, Gather);

}  // namespace intel_gpu
}  // namespace ov
