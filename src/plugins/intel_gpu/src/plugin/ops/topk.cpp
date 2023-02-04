// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/topk.hpp"

#include "intel_gpu/primitives/arg_max_min.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

namespace ov {
namespace intel_gpu {

static void CreateTopKOp(Program& p, const std::shared_ptr<ngraph::op::v1::TopK>& op) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    ov::op::TopKMode mode = op->get_mode();
    ov::op::TopKSortType stype = op->get_sort_type();

    uint32_t top_k = op->get_k();
    uint64_t chosen_axis = op->get_axis();

    if (p.use_new_shape_infer()) {
        size_t num_outputs = op->get_output_size();
        auto get_output_paddings = [&]() {
            std::vector<cldnn::padding> output_paddings;
            for (size_t i = 0; i < num_outputs; i++)
                output_paddings.push_back(cldnn::padding());
            return output_paddings;
        };
        auto get_output_data_types = [&]() {
            std::vector<cldnn::optional_data_type> output_data_types;
            for (size_t i = 0; i < num_outputs; i++) {
                auto type = op->get_output_element_type(i);
                output_data_types.push_back(cldnn::element_type_to_data_type(type));
            }
            return output_data_types;
        };
        auto argmaxPrim = cldnn::arg_max_min(layerName,
                                             inputs,
                                             mode,
                                             top_k,
                                             chosen_axis,
                                             stype,
                                             true,
                                             cldnn::padding({0, 0, 0, 0}, 0),
                                             cldnn::element_type_to_data_type(op->get_output_element_type(0)),
                                             num_outputs);
        argmaxPrim.output_paddings = get_output_paddings();
        argmaxPrim.output_data_types = get_output_data_types();
        p.add_primitive(*op, argmaxPrim);
    } else {
        if (op->get_output_size() == 2) {
            auto mutable_precision = op->get_output_element_type(1);
            if (mutable_precision == ngraph::element::i64) {
                mutable_precision = ngraph::element::i32;
            }

            cldnn::layout mutableLayout = cldnn::layout(cldnn::element_type_to_data_type(mutable_precision),
                                                        cldnn::format::get_default_format(op->get_output_shape(1).size()),
                                                        tensor_from_dims(op->get_output_shape(1)));

            GPU_DEBUG_LOG << "[" << layer_type_name_ID(op) << ": mutable data]" << std::endl;
            auto shared_memory = p.get_engine().allocate_memory(mutableLayout);

            cldnn::primitive_id argmax_mutable_id_w = layer_type_name_ID(op) + "_md_write";
            auto argmax_mutable_prim = cldnn::mutable_data(argmax_mutable_id_w,
                                                           shared_memory);
            p.add_primitive(*op, argmax_mutable_prim);
            inputs.push_back(cldnn::input_info(argmax_mutable_id_w));

            std::string ArgMaxLayerName = layerName + ".out0";
            auto argmaxPrim = cldnn::arg_max_min(ArgMaxLayerName,
                                                 inputs,
                                                 mode,
                                                 top_k,
                                                 chosen_axis,
                                                 stype,
                                                 true,
                                                 cldnn::padding({0, 0, 0, 0}, 0),
                                                 cldnn::element_type_to_data_type(op->get_output_element_type(0)));

            p.add_primitive(*op, argmaxPrim);

            cldnn::primitive_id argmax_mutable_id_r = layerName + ".out1";
            auto argmax_mutable_prim_r = cldnn::mutable_data(argmax_mutable_id_r,
                                                             { cldnn::input_info(ArgMaxLayerName) },
                                                             shared_memory);
            p.add_primitive(*op, argmax_mutable_prim_r);
        } else if (op->get_output_size() == 1) {
            auto argmaxPrim = cldnn::arg_max_min(layerName,
                                                 inputs,
                                                 mode,
                                                 top_k,
                                                 chosen_axis,
                                                 stype,
                                                 true,
                                                 cldnn::padding({0, 0, 0, 0}, 0),
                                                 cldnn::element_type_to_data_type(op->get_output_element_type(0)));

            p.add_primitive(*op, argmaxPrim);
        } else {
            IE_THROW() << op->get_friendly_name() << " Incorrect TopK outputs number";
        }
    }
}

REGISTER_FACTORY_IMPL(v1, TopK);

}  // namespace intel_gpu
}  // namespace ov
