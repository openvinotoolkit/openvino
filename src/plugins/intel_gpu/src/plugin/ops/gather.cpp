// Copyright (C) 2018-2023 Intel Corporation
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
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    int64_t axis = op->get_axis();

    std::vector<cldnn::input_info> reordered_inputs;
    reordered_inputs.resize(inputs.size());

    for (size_t portIndex = 0; portIndex < inputs.size(); portIndex++) {
        auto inputDataType = cldnn::element_type_to_data_type(op->get_input_element_type(portIndex));
        if (inputDataType == cldnn::data_types::i64) {
            // GPU primitive does not support i64 inputs,
            // so we need additional reorders to convert them to i32
            auto reorderPrimName = inputs[portIndex].pid + "_" + op->get_friendly_name() + Program::m_preProcessTag;
            auto targetFormat = cldnn::format::get_default_format(op->get_input_shape(portIndex).size());
            auto preprocessPrim = cldnn::reorder(reorderPrimName,
                                                 inputs[portIndex],
                                                 targetFormat,
                                                 cldnn::data_types::i32,
                                                 std::vector<float>(),
                                                 cldnn::reorder_mean_mode::subtract);
            p.add_primitive(*op, preprocessPrim);
            reordered_inputs[portIndex] = cldnn::input_info(reorderPrimName);
        } else {
            reordered_inputs[portIndex] = inputs[portIndex];
        }
    }

    // Dynamic path will do shape infer internally, so no need to pass valid out shape for that case
    ov::Shape out_shape = op->get_output_partial_shape(0).is_static() ? op->get_output_shape(0) : ov::Shape{};

    auto gatherPrim = cldnn::gather(layerName,
                                    reordered_inputs[0],
                                    reordered_inputs[1],
                                    axis,
                                    out_shape,
                                    batch_dim,
                                    support_neg_ind);

    p.add_primitive(*op, gatherPrim);
}

static void CreateGatherOp(Program& p, const std::shared_ptr<ngraph::op::v1::Gather>& op) {
    validate_inputs_count(op, {2, 3});
    CreateGatherOpBase<ngraph::op::v1::Gather>(p, op);
}

REGISTER_FACTORY_IMPL(v1, Gather);

static void CreateGatherOp(Program& p, const std::shared_ptr<ngraph::op::v7::Gather>& op) {
    validate_inputs_count(op, {2, 3, 4});
    CreateGatherOpBase<ngraph::op::v7::Gather>(p, op, op->get_batch_dims());
}

REGISTER_FACTORY_IMPL(v7, Gather);

static void CreateGatherOp(Program& p, const std::shared_ptr<ngraph::op::v8::Gather>& op) {
    validate_inputs_count(op, {2, 3, 4});
    CreateGatherOpBase<ngraph::op::v8::Gather>(p, op, op->get_batch_dims(), true);
}

REGISTER_FACTORY_IMPL(v8, Gather);

}  // namespace intel_gpu
}  // namespace ov
