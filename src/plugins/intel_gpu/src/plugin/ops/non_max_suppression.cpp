// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/non_max_suppression.hpp"
#include <ov_ops/nms_ie_internal.hpp>

#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/non_max_suppression.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

namespace ov::intel_gpu {
namespace {
    static std::vector<cldnn::input_info> reorder_nms_inputs_if_needed(
    ProgramBuilder& p,
    const std::shared_ptr<ov::Node>& op,
    const std::vector<cldnn::input_info>& inputs) {

    std::vector<cldnn::input_info> reordered(inputs.size());

    for (size_t portIndex = 0; portIndex < inputs.size(); portIndex++) {
        auto inputDataType = cldnn::element_type_to_data_type(op->get_input_element_type(portIndex));

        if (portIndex == 2 && inputDataType == cldnn::data_types::i64) {
            // Convert i64 to i32
            auto reorder_id = inputs[portIndex].pid + "_" + op->get_friendly_name() + ProgramBuilder::m_preProcessTag;
            auto fmt = cldnn::format::get_default_format(op->get_input_partial_shape(portIndex).size());
            auto reorder = cldnn::reorder(reorder_id, inputs[portIndex], fmt, cldnn::data_types::i32);
            p.add_primitive(*op, reorder);
            reordered[portIndex] = cldnn::input_info(reorder_id);
        } else {
            reordered[portIndex] = inputs[portIndex];
        }
    }

    return reordered;
}

static void apply_optional_inputs(const std::shared_ptr<ov::Node>& op, cldnn::non_max_suppression& prim, const std::vector<cldnn::input_info>& inputs) {
    switch (inputs.size()) {
        case 6: prim.soft_nms_sigma = inputs[5].pid; [[fallthrough]];
        case 5: prim.score_threshold = inputs[4].pid; [[fallthrough]];
        case 4: prim.iou_threshold = inputs[3].pid; [[fallthrough]];
        case 3: prim.num_select_per_class = inputs[2].pid; [[fallthrough]];
        case 2: break;
        default:
            OPENVINO_THROW("Incorrect number of input primitives for layer: ", op->get_friendly_name());
    }
}

static void add_nms_gather_if_needed(ProgramBuilder& p, const std::shared_ptr<ov::Node>& op, const std::string& nms_name, size_t num_outputs) {
    if (!op->get_output_partial_shape(0).is_dynamic())
        return;

    auto gather_name = nms_name + "_NMSGather";
    std::vector<cldnn::input_info> gather_inputs = {
        cldnn::input_info(nms_name, 0),
        cldnn::input_info(nms_name, 1),
        cldnn::input_info(nms_name, 2)
    };

    gather_inputs.resize(num_outputs);  // Only take as many as outputs

    auto gather_prim = cldnn::non_max_suppression_gather(gather_name, gather_inputs, num_outputs);
    p.add_primitive(*op, gather_prim);
}
} // namespace

static void CreateNonMaxSuppressionIEInternalOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::NonMaxSuppressionIEInternal>& op) {
    cldnn::non_max_suppression::Rotation rotation = cldnn::non_max_suppression::Rotation::NONE;
    const bool is_nms_rotated = op->m_rotation != ov::op::internal::NonMaxSuppressionIEInternal::Rotation_None;
    if (is_nms_rotated) {
        // For NMSRotated threshold inputs are mandatory, and soft_nms_sigma input is absent
        validate_inputs_count(op, {5});

        rotation = op->m_rotation == ov::op::internal::NonMaxSuppressionIEInternal::Rotation_Clockwise ?
                    cldnn::non_max_suppression::Rotation::CLOCKWISE
                    : cldnn::non_max_suppression::Rotation::COUNTERCLOCKWISE;
    } else {
        validate_inputs_count(op, {2, 3, 4, 5, 6});
    }

    auto inputs = p.GetInputInfo(op);
    auto reordered_inputs = reorder_nms_inputs_if_needed(p, op, inputs);
    size_t num_outputs = op->get_output_size();
    std::string nms_name = layer_type_name_ID(op);

    if (p.use_new_shape_infer()) {
        auto prim = cldnn::non_max_suppression(
                    nms_name,
                    reordered_inputs[0],
                    reordered_inputs[1],
                    0,
                    op->m_center_point_box,
                    op->m_sort_result_descending,
                    "", "", "", "", "", "", num_outputs);

        prim.output_data_types = get_output_data_types(op, {{ov::element::i64, ov::element::i32}});
        prim.rotation = rotation;

        apply_optional_inputs(op, prim, reordered_inputs);
        p.add_primitive(*op, prim);
        add_nms_gather_if_needed(p, op, nms_name, num_outputs);

    } else {
        auto outputIndices = op->get_output_partial_shape(0)[0].get_length();

        std::vector<cldnn::memory::ptr> shared_memory;
        switch (num_outputs) {
            case 3: {
                auto mutable_precision_second = op->get_output_element_type(2);
                if (mutable_precision_second == ov::element::i64) {
                    mutable_precision_second = ov::element::i32;
                }
                cldnn::layout mutableLayoutSecond = cldnn::layout(
                    cldnn::element_type_to_data_type(mutable_precision_second),
                    cldnn::format::get_default_format(op->get_output_shape(2).size()),
                    tensor_from_dims(op->get_output_shape(2)));

                GPU_DEBUG_LOG << "[" << layer_type_name_ID(op) << ": mutable data]" << std::endl;
                shared_memory.emplace_back(p.get_engine().allocate_memory(mutableLayoutSecond));

                cldnn::primitive_id non_max_suppression_mutable_id_w_second = layer_type_name_ID(op) + "_md_write_second";
                auto nms_mutable_prim_second = cldnn::mutable_data(non_max_suppression_mutable_id_w_second,
                                                                   shared_memory.back());
                p.add_primitive(*op, nms_mutable_prim_second);
                inputs.push_back(cldnn::input_info(non_max_suppression_mutable_id_w_second));
            }
            case 2: {
                auto mutable_precision_first = op->get_output_element_type(1);
                cldnn::layout mutableLayoutFirst = cldnn::layout(
                    cldnn::element_type_to_data_type(mutable_precision_first),
                    cldnn::format::bfyx,
                    cldnn::tensor(static_cast<int32_t>(outputIndices), 3, 1, 1));

                GPU_DEBUG_LOG << "[" << layer_type_name_ID(op) << ": mutable data]" << std::endl;
                shared_memory.emplace_back(p.get_engine().allocate_memory(mutableLayoutFirst));

                cldnn::primitive_id non_max_suppression_mutable_id_w_first = layer_type_name_ID(op) + "_md_write_first";
                auto nms_mutable_prim_first = cldnn::mutable_data(non_max_suppression_mutable_id_w_first,
                                                                  shared_memory.back());
                p.add_primitive(*op, nms_mutable_prim_first);
                inputs.push_back(cldnn::input_info(non_max_suppression_mutable_id_w_first));
            }
            case 1: break;
            default: OPENVINO_THROW("Incorrect number of output for layer: ", op->get_friendly_name());
        }

        auto nonMaxSuppressionLayerName = num_outputs > 1 ? layer_type_name_ID(op) + ".out0" : layer_type_name_ID(op);

        auto prim = cldnn::non_max_suppression(
                    nonMaxSuppressionLayerName,
                    reordered_inputs[0],
                    reordered_inputs[1],
                    static_cast<int>(outputIndices),
                    op->m_center_point_box,
                    op->m_sort_result_descending,
                    "", "", "", "", "", "");

        prim.output_data_types = get_output_data_types(op, {{ov::element::i64, ov::element::i32}});
        prim.rotation = rotation;

        apply_optional_inputs(op, prim, reordered_inputs);

        switch (num_outputs) {
            case 3: prim.third_output = inputs[inputs.size() - 2].pid;
            case 2: prim.second_output = inputs[inputs.size() - 1].pid;
            default: break;
        }

        p.add_primitive(*op, prim);

        switch (num_outputs) {
            case 3: {
                cldnn::primitive_id non_max_suppression_id_r_second = layer_type_name_ID(op) + ".out2";
                auto nms_mutable_prim_r_second = cldnn::mutable_data(non_max_suppression_id_r_second,
                                                                     { cldnn::input_info(nonMaxSuppressionLayerName) },
                                                                     shared_memory.front());
                p.add_primitive(*op, nms_mutable_prim_r_second);
            }
            case 2: {
                cldnn::primitive_id non_max_suppression_id_r_first = layer_type_name_ID(op) + ".out1";
                auto nms_mutable_prim_r_first = cldnn::mutable_data(non_max_suppression_id_r_first,
                                                                    { cldnn::input_info(nonMaxSuppressionLayerName) },
                                                                    shared_memory.back());
                p.add_primitive(*op, nms_mutable_prim_r_first);
            }
            default: break;
        }
    }
}

static void CreateNonMaxSuppressionOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v9::NonMaxSuppression>& op) {
    validate_inputs_count(op, {2, 3, 4, 5, 6});
    auto inputs = p.GetInputInfo(op);
    auto reordered_inputs = reorder_nms_inputs_if_needed(p, op, inputs);
    size_t num_outputs = op->get_output_size();
    std::string nms_name = layer_type_name_ID(op);

    bool center_point_box = (op->get_box_encoding() == ov::op::v9::NonMaxSuppression::BoxEncodingType::CENTER);

    auto prim = cldnn::non_max_suppression(
                nms_name,
                reordered_inputs[0],
                reordered_inputs[1],
                0,
                center_point_box,
                op->get_sort_result_descending(),
                "", "", "", "", "", "", num_outputs);

    prim.output_data_types = get_output_data_types(op, {{ov::element::i64, ov::element::i32}});
    prim.rotation = cldnn::non_max_suppression::Rotation::NONE;

    apply_optional_inputs(op, prim, reordered_inputs);
    p.add_primitive(*op, prim);
    add_nms_gather_if_needed(p, op, nms_name, num_outputs);
}

REGISTER_FACTORY_IMPL(internal, NonMaxSuppressionIEInternal);
REGISTER_FACTORY_IMPL(v9, NonMaxSuppression);
}  // namespace ov::intel_gpu
