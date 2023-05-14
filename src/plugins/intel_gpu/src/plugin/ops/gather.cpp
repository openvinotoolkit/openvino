// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/gather.hpp"

#include "intel_gpu/primitives/gather.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/reshape.hpp"

using namespace InferenceEngine;
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
            auto targetFormat = cldnn::format::get_default_format(op->get_input_partial_shape(portIndex).size());
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

    // Dynamic path will do shape infer internally, so no need to pass valid out shape for that case
    ov::Shape out_shape = op->get_output_partial_shape(0).is_static() ? op->get_output_shape(0) : ov::Shape{};

    // Update output_shape in case of scalar indice
    bool need_reshape = false;
    auto out_shape_original = out_shape;
    if (!p.use_new_shape_infer() && op->get_output_partial_shape(0).is_static()) {
        auto input1_shape = op->get_input_shape(1);
        if (input1_shape.size() == 0 && batch_dim == 0) {
            need_reshape = true;

            auto new_axis = axis;
            if (new_axis < 0) {
                new_axis += op->get_input_shape(0).size();
            }
            out_shape.push_back(1);
            for (int i = static_cast<int>(out_shape.size()) - 1; i > new_axis ; i--) {
                out_shape[i] = out_shape[i-1];
            }
            out_shape[new_axis] = 1;
        }
    }

    // WA for NMS->Gather construction. NMS fills part of the output blob by the -1 if these values
    // must not be taken into account.
    // CPU also uses this like of WA.
    if (support_neg_ind) {
        const auto& rti = op->get_rt_info();
        const auto& reverse = rti.find("dontReverseIndices");
        if (reverse != rti.end()) {
            support_neg_ind = false;
        }
    }

    // gather
    auto reshapeName = layerName + "";
    if (need_reshape) {
        layerName = layerName + "_reshape_output";
    }

    auto gatherPrim = cldnn::gather(layerName,
                                    reordered_inputs[0],
                                    reordered_inputs[1],
                                    axis,
                                    out_shape,
                                    batch_dim,
                                    support_neg_ind);

    p.add_primitive(*op, gatherPrim);

    // Add reorder and reshape for scalar indice
    if (need_reshape) {
        auto input = inputs[0];
        input.pid = layerName;

        auto targetFormat = cldnn::format::get_default_format(out_shape_original.size());
        if (targetFormat.value != cldnn::format::get_default_format(out_shape.size()).value) {
            auto reorderName = layerName + "_cldnn_in_reorder";
            auto targetDatatype = cldnn::element_type_to_data_type(op->get_input_element_type(0));
            auto reorderPrim = cldnn::reorder(reorderName,
                                              input,
                                              targetFormat,
                                              targetDatatype);
            p.add_primitive(*op, reorderPrim);
            input.pid = reorderName;
        }

        auto reshapePrim = cldnn::reshape(reshapeName, input, tensor_from_dims(out_shape_original));
        p.add_primitive(*op, reshapePrim);
    }
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
