// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/non_max_suppression.hpp"
#include <ngraph/opsets/opset3.hpp>
#include <ov_ops/nms_ie_internal.hpp>

#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/non_max_suppression.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

namespace ov {
namespace intel_gpu {

static void CreateNonMaxSuppressionIEInternalOp(Program& p, const std::shared_ptr<ngraph::op::internal::NonMaxSuppressionIEInternal>& op) {
    validate_inputs_count(op, {2, 3, 4, 5, 6});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::vector<cldnn::primitive_id> reorderedInputs;
    reorderedInputs.resize(inputPrimitives.size());

    for (size_t portIndex = 0; portIndex < inputPrimitives.size(); portIndex++) {
        auto inputDataType = cldnn::element_type_to_data_type(op->get_input_element_type(portIndex));
        if ((portIndex == 2) && (inputDataType == cldnn::data_types::i64)) {
            // GPU primitive supports only i32 data type for 'max_output_boxes_per_class' input
            // so we need additional reorder if it's provided as i64
            auto reorderPrimName = inputPrimitives[portIndex] + "_" + op->get_friendly_name() + Program::m_preProcessTag;
            auto targetFormat = cldnn::format::get_default_format(op->get_input_partial_shape(portIndex).size());
            auto preprocessPrim = cldnn::reorder(reorderPrimName,
                                                 inputPrimitives[portIndex],
                                                 targetFormat,
                                                 cldnn::data_types::i32,
                                                 std::vector<float>(),
                                                 cldnn::reorder_mean_mode::subtract);
            p.add_primitive(*op, preprocessPrim);
            reorderedInputs[portIndex] = (reorderPrimName);
        } else {
            reorderedInputs[portIndex] = inputPrimitives[portIndex];
        }
    }

    // GPU primitive supports only i32 as output data type
    auto out_type = op->get_output_element_type(0);
    if (out_type == ngraph::element::i64) {
        out_type = ngraph::element::i32;
    }

    auto boxesShape = op->get_input_partial_shape(0);
    std::size_t num_output = op->get_output_size();

    if (p.use_new_shape_infer()) {
        std::vector<cldnn::input_info> inputs;
        for (size_t i = 0; i < 2; ++i) {
            inputs.push_back(cldnn::input_info(reorderedInputs[i], op->get_input_source_output(i).get_index()));
        }
        auto nonMaxSupressionLayerName = layer_type_name_ID(op);
        auto prim = cldnn::non_max_suppression(
                nonMaxSupressionLayerName,
                reorderedInputs[0],
                reorderedInputs[1],
                0,
                op->m_center_point_box,
                op->m_sort_result_descending,
                "", "", "", "", "", "", inputs, num_output);

        prim.output_data_type = cldnn::element_type_to_data_type(out_type);

        switch (reorderedInputs.size()) {
            case 6: prim.soft_nms_sigma = reorderedInputs[5];
            case 5: prim.score_threshold = reorderedInputs[4];
            case 4: prim.iou_threshold = reorderedInputs[3];
            case 3: prim.num_select_per_class = reorderedInputs[2];
            case 2: break;
            default: IE_THROW() << "Incorrect number of input primitives for layer: " << op->get_friendly_name();
        }

        p.add_primitive(*op, prim);
    } else {
        auto outputIndices = op->get_output_partial_shape(0)[0].get_length();

        std::vector<cldnn::memory::ptr> shared_memory;
        GPU_DEBUG_GET_INSTANCE(debug_config);
        switch (num_output) {
            case 3: {
                auto mutable_precision_second = op->get_output_element_type(2);
                if (mutable_precision_second == ngraph::element::i64) {
                    mutable_precision_second = ngraph::element::i32;
                }
                cldnn::layout mutableLayoutSecond = cldnn::layout(
                    cldnn::element_type_to_data_type(mutable_precision_second),
                    cldnn::format::get_default_format(op->get_output_shape(2).size()),
                    tensor_from_dims(op->get_output_shape(2)));

                GPU_DEBUG_IF(debug_config->verbose >= 2) {
                    GPU_DEBUG_COUT << "[" << layer_type_name_ID(op) << ": mutable data]" << std::endl;
                }
                shared_memory.emplace_back(p.GetEngine().allocate_memory(mutableLayoutSecond));

                cldnn::primitive_id non_max_supression_mutable_id_w_second = layer_type_name_ID(op) + "_md_write_second";
                auto nms_mutable_prim_second = cldnn::mutable_data(non_max_supression_mutable_id_w_second,
                                                                   shared_memory.back());
                p.add_primitive(*op, nms_mutable_prim_second);
                inputPrimitives.push_back(non_max_supression_mutable_id_w_second);
            }
            case 2: {
                auto mutable_precision_first = op->get_output_element_type(1);
                cldnn::layout mutableLayoutFirst = cldnn::layout(
                    cldnn::element_type_to_data_type(mutable_precision_first),
                    cldnn::format::bfyx,
                    cldnn::tensor(static_cast<int32_t>(outputIndices), 3, 1, 1));

                GPU_DEBUG_IF(debug_config->verbose >= 2) {
                    GPU_DEBUG_COUT << "[" << layer_type_name_ID(op) << ": mutable data]" << std::endl;
                }
                shared_memory.emplace_back(p.GetEngine().allocate_memory(mutableLayoutFirst));

                cldnn::primitive_id non_max_supression_mutable_id_w_first = layer_type_name_ID(op) + "_md_write_first";
                auto nms_mutable_prim_first = cldnn::mutable_data(non_max_supression_mutable_id_w_first,
                                                                  shared_memory.back());
                p.add_primitive(*op, nms_mutable_prim_first);
                inputPrimitives.push_back(non_max_supression_mutable_id_w_first);
            }
            case 1: break;
            default: IE_THROW() << "Incorrect number of output for layer: " << op->get_friendly_name();
        }

        auto nonMaxSupressionLayerName = num_output > 1 ? layer_type_name_ID(op) + ".out0" : layer_type_name_ID(op);

        auto prim = cldnn::non_max_suppression(
                nonMaxSupressionLayerName,
                reorderedInputs[0],
                reorderedInputs[1],
                static_cast<int>(outputIndices),
                op->m_center_point_box,
                op->m_sort_result_descending,
                "", "", "", "", "", "");

        prim.output_data_type = cldnn::element_type_to_data_type(out_type);

        switch (reorderedInputs.size()) {
            case 6: prim.soft_nms_sigma = reorderedInputs[5];
            case 5: prim.score_threshold = reorderedInputs[4];
            case 4: prim.iou_threshold = reorderedInputs[3];
            case 3: prim.num_select_per_class = reorderedInputs[2];
            case 2: break;
            default: IE_THROW() << "Incorrect number of input primitives for layer: " << op->get_friendly_name();
        }

        switch (num_output) {
            case 3: prim.third_output = inputPrimitives[inputPrimitives.size() - 2];
            case 2: prim.second_output = inputPrimitives[inputPrimitives.size() - 1];
            default: break;
        }

        p.add_primitive(*op, prim);

        switch (num_output) {
            case 3: {
                cldnn::primitive_id non_max_supression_id_r_second = layer_type_name_ID(op) + ".out2";
                auto nms_mutable_prim_r_second = cldnn::mutable_data(non_max_supression_id_r_second,
                                                                     { nonMaxSupressionLayerName },
                                                                     shared_memory.front());
                p.add_primitive(*op, nms_mutable_prim_r_second);
            }
            case 2: {
                cldnn::primitive_id non_max_supression_id_r_first = layer_type_name_ID(op) + ".out1";
                auto nms_mutable_prim_r_first = cldnn::mutable_data(non_max_supression_id_r_first,
                                                                    { nonMaxSupressionLayerName },
                                                                    shared_memory.back());
                p.add_primitive(*op, nms_mutable_prim_r_first);
            }
            default: break;
        }
    }
}

REGISTER_FACTORY_IMPL(internal, NonMaxSuppressionIEInternal);

}  // namespace intel_gpu
}  // namespace ov
