// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/multiclass_nms.hpp>
#include "ov_ops/multiclass_nms_ie_internal.hpp"

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/multiclass_nms.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"

namespace ov::intel_gpu {


static void CreateMulticlassNmsIEInternalOp(ProgramBuilder& p, const std::shared_ptr<op::internal::MulticlassNmsIEInternal>& op) {
    validate_inputs_count(op, {2, 3});

    auto inputs = p.GetInputInfo(op);

    if (p.use_new_shape_infer()) {
        cldnn::multiclass_nms prim{layer_type_name_ID(op), inputs, op->get_attrs()};
        prim.num_outputs = op->get_output_size();
        prim.output_data_types = get_output_data_types(op, {{ov::element::i64, ov::element::i32}});

        p.add_primitive(*op, prim);
    } else {
        if (inputs.size() == 2) {
            inputs.push_back(cldnn::input_info(""));  // roisnum dummy id
        }

        const auto op_friendly_name = op->get_friendly_name();

        const auto layer_type_name = layer_type_name_ID(op);
        const auto layer_name = layer_type_name + ".out0";

        const auto mutable_precision1 = op->get_output_element_type(1);
        const auto output_shape1 = op->get_output_shape(1);
        const cldnn::layout mutable_layout1{cldnn::element_type_to_data_type(mutable_precision1),
                                            cldnn::format::get_default_format(output_shape1.size()),
                                            tensor_from_dims(output_shape1)};
        cldnn::memory::ptr shared_memory1{p.get_engine().allocate_memory(mutable_layout1)};

        const auto mutable_id_w1 = layer_type_name + "_md_write.1";
        const cldnn::mutable_data mutable_prim_w{mutable_id_w1, shared_memory1};
        p.add_primitive(*op, mutable_prim_w);
        inputs.push_back(cldnn::input_info(mutable_id_w1));

        const auto mutable_precision2 = op->get_output_element_type(2);
        const auto output_shape2 = op->get_output_shape(2);
        const cldnn::layout mutable_layout2{cldnn::element_type_to_data_type(mutable_precision2),
                                            cldnn::format::get_default_format(output_shape2.size()),
                                            tensor_from_dims(output_shape2)};
        cldnn::memory::ptr shared_memory2{p.get_engine().allocate_memory(mutable_layout2)};

        const auto mutable_id_w2 = layer_type_name + "_md_write.2";
        const cldnn::mutable_data mutable_prim_w2{mutable_id_w2, shared_memory2};
        p.add_primitive(*op, mutable_prim_w2);
        inputs.push_back(cldnn::input_info(mutable_id_w2));

        constexpr auto expected_inputs_count = 3 + 2;  // 3 operation inputs plus 2 additional outputs
        if (inputs.size() != expected_inputs_count) {
            OPENVINO_THROW("multiclass_nms primitive requires 5 inputs");
        }

        const cldnn::multiclass_nms prim{layer_name,
                                        inputs,
                                        op->get_attrs()};

        p.add_primitive(*op, prim);

        const auto mutable_id_r1 = layer_type_name + ".out1";
        const cldnn::mutable_data mutable_prim_r1{mutable_id_r1, {cldnn::input_info(layer_name)}, shared_memory1};
        p.add_primitive(*op, mutable_prim_r1);

        const auto mutable_id_r2 = layer_type_name + ".out2";
        const cldnn::mutable_data mutable_prim_r2{mutable_id_r2, {cldnn::input_info(layer_name)}, shared_memory2};
        p.add_primitive(*op, mutable_prim_r2);
    }
}

REGISTER_FACTORY_IMPL(internal, MulticlassNmsIEInternal);

}  // namespace ov::intel_gpu
