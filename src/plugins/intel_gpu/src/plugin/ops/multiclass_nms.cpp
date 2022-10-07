// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/multiclass_nms.hpp>
#include "ngraph_ops/multiclass_nms_ie_internal.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/primitives/multiclass_nms.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"

namespace ov {
namespace intel_gpu {

static cldnn::sort_result_type GetSortResultType(const ngraph::op::util::MulticlassNmsBase::SortResultType sort_result_type) {
    switch (sort_result_type) {
        case ngraph::op::util::MulticlassNmsBase::SortResultType::CLASSID:
            return cldnn::sort_result_type::classid;
        case ngraph::op::util::MulticlassNmsBase::SortResultType::SCORE:
            return cldnn::sort_result_type::score;
        case ngraph::op::util::MulticlassNmsBase::SortResultType::NONE:
            return cldnn::sort_result_type::none;
        default: IE_THROW() << "Unsupported SortResultType value: " << static_cast<int>(sort_result_type);
    }
    return cldnn::sort_result_type::none;
}

static void CreateMulticlassNmsIEInternalOp(Program& p, const std::shared_ptr<ngraph::op::internal::MulticlassNmsIEInternal>& op) {
    validate_inputs_count(op, {2, 3});

    if (op->get_output_size() != 3) {
        IE_THROW() << "MulticlassNms requires 3 outputs";
    }

    auto inputs = p.GetInputPrimitiveIDs(op);
    if (inputs.size() == 2) {
        inputs.push_back("");  // roisnum dummy id
    }

    const auto& attrs = op->get_attrs();

    const auto op_friendly_name = op->get_friendly_name();

    const auto layer_type_name = layer_type_name_ID(op);
    const auto layer_name = layer_type_name + ".out0";

    const auto mutable_precision1 = op->get_output_element_type(1);
    const auto output_shape1 = op->get_output_shape(1);
    const cldnn::layout mutable_layout1{cldnn::element_type_to_data_type(mutable_precision1),
                                        cldnn::format::get_default_format(output_shape1.size()),
                                        tensor_from_dims(output_shape1)};
    cldnn::memory::ptr shared_memory1{p.GetEngine().allocate_memory(mutable_layout1)};

    const auto mutable_id_w1 = layer_type_name + "_md_write.1";
    const cldnn::mutable_data mutable_prim_w{mutable_id_w1, shared_memory1};
    p.add_primitive(*op, mutable_prim_w);
    inputs.push_back(mutable_id_w1);

    const auto mutable_precision2 = op->get_output_element_type(2);
    const auto output_shape2 = op->get_output_shape(2);
    const cldnn::layout mutable_layout2{cldnn::element_type_to_data_type(mutable_precision2),
                                        cldnn::format::get_default_format(output_shape2.size()),
                                        tensor_from_dims(output_shape2)};
    cldnn::memory::ptr shared_memory2{p.GetEngine().allocate_memory(mutable_layout2)};

    const auto mutable_id_w2 = layer_type_name + "_md_write.2";
    const cldnn::mutable_data mutable_prim_w2{mutable_id_w2, shared_memory2};
    p.add_primitive(*op, mutable_prim_w2);
    inputs.push_back(mutable_id_w2);

    constexpr auto expected_inputs_count = 3 + 2;  // 3 operation inputs plus 2 additional outputs
    if (inputs.size() != expected_inputs_count) {
        IE_THROW() << "multiclass_nms primitive requires 5 inputs";
    }

    const auto sort_result_type = GetSortResultType(attrs.sort_result_type);

    const cldnn::multiclass_nms prim{layer_name,
                                     inputs[0],
                                     inputs[1],
                                     inputs[2],
                                     inputs[3],
                                     inputs[4],
                                     sort_result_type,
                                     attrs.sort_result_across_batch,
                                     cldnn::element_type_to_data_type(attrs.output_type),
                                     attrs.iou_threshold,
                                     attrs.score_threshold,
                                     attrs.nms_top_k,
                                     attrs.keep_top_k,
                                     attrs.background_class,
                                     attrs.normalized,
                                     attrs.nms_eta,
                                     op_friendly_name};

    p.add_primitive(*op, prim);

    const auto mutable_id_r1 = layer_type_name + ".out1";
    const cldnn::mutable_data mutable_prim_r1{mutable_id_r1, {layer_name}, shared_memory1};
    p.add_primitive(*op, mutable_prim_r1);

    const auto mutable_id_r2 = layer_type_name + ".out2";
    const cldnn::mutable_data mutable_prim_r2{mutable_id_r2, {layer_name}, shared_memory2};
    p.add_primitive(*op, mutable_prim_r2);
}

REGISTER_FACTORY_IMPL(internal, MulticlassNmsIEInternal);

}  // namespace intel_gpu
}  // namespace ov
