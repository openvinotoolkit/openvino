// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/unique.hpp"

#include "intel_gpu/plugin/program.hpp"
#include "ngraph/op/unique.hpp"

namespace ov {
namespace intel_gpu {

namespace {

void CreateUniqueOp(Program& p, const std::shared_ptr<ngraph::op::v10::Unique>& op) {
    validate_inputs_count(op, {1, 2});

    bool flattened = true;
    int64_t axis{};
    if (op->get_input_size() == 2) {
        auto axis_constant = std::dynamic_pointer_cast<ngraph::op::Constant>(op->get_input_node_shared_ptr(1));
        if (!axis_constant) {
            IE_THROW() << "Unsupported parameter nodes type in " << op->get_friendly_name() << " ("
                       << op->get_type_name() << ")";
        }
        axis = axis_constant->cast_vector<int64_t>().at(0);
        axis = ov::normalize_axis(op.get(), axis, op->get_input_partial_shape(0).rank());
        flattened = false;
    }

    std::vector<cldnn::optional_data_type> output_data_types;
    output_data_types.emplace_back(cldnn::element_type_to_data_type(op->get_count_element_type()));
    output_data_types.emplace_back(cldnn::element_type_to_data_type(op->get_input_element_type(0)));
    output_data_types.emplace_back(cldnn::element_type_to_data_type(op->get_index_element_type()));
    output_data_types.emplace_back(cldnn::element_type_to_data_type(op->get_index_element_type()));
    output_data_types.emplace_back(cldnn::element_type_to_data_type(op->get_count_element_type()));
    const auto num_outputs = output_data_types.size();
    const std::vector<cldnn::padding> output_paddings(num_outputs);

    const auto layer_name = layer_type_name_ID(op);
    const cldnn::unique unique_prim(layer_name,
                                    p.GetInputInfo(op).at(0),
                                    flattened,
                                    axis,
                                    op->get_sorted(),
                                    output_paddings,
                                    output_data_types,
                                    num_outputs);
    p.add_primitive(*op, unique_prim);

    // We add unique reshape primitive to adjust outputs shapes to founded unique count
    std::vector<cldnn::optional_data_type> output_data_types_reshape;
    for (auto i = 1U; i < output_data_types.size(); ++i) {
        output_data_types_reshape.emplace_back(output_data_types.at(i));
    }
    const auto num_outputs_reshape = output_data_types_reshape.size();
    const std::vector<cldnn::padding> output_paddings_reshape(num_outputs_reshape);

    const auto layer_name_reshape = layer_name + "_reshape";
    const cldnn::unique_reshape unique_reshape_prim(layer_name_reshape,
                                                    {cldnn::input_info(layer_name, 0),
                                                     cldnn::input_info(layer_name, 1),
                                                     cldnn::input_info(layer_name, 2),
                                                     cldnn::input_info(layer_name, 3),
                                                     cldnn::input_info(layer_name, 4)},
                                                    flattened,
                                                    axis,
                                                    output_paddings_reshape,
                                                    output_data_types_reshape,
                                                    num_outputs_reshape);
    p.add_primitive(*op, unique_reshape_prim);
}

}  // namespace

REGISTER_FACTORY_IMPL(v10, Unique);

}  // namespace intel_gpu
}  // namespace ov
