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
        flattened = false;
    }

    const std::vector<cldnn::padding> output_paddings(5);
    std::vector<cldnn::optional_data_type> output_data_types;
    output_data_types.emplace_back(cldnn::element_type_to_data_type(op->get_input_element_type(0)));
    output_data_types.emplace_back(cldnn::element_type_to_data_type(op->get_index_element_type()));
    output_data_types.emplace_back(cldnn::element_type_to_data_type(op->get_index_element_type()));
    output_data_types.emplace_back(cldnn::element_type_to_data_type(op->get_count_element_type()));
    output_data_types.emplace_back(cldnn::element_type_to_data_type(op->get_count_element_type()));

    const cldnn::unique unique_prim(layer_type_name_ID(op),
                                    p.GetInputInfo(op).at(0),
                                    flattened,
                                    axis,
                                    op->get_sorted(),
                                    true,
                                    output_paddings,
                                    output_data_types,
                                    5);
    p.add_primitive(*op, unique_prim);
}

}  // namespace

REGISTER_FACTORY_IMPL(v10, Unique);

}  // namespace intel_gpu
}  // namespace ov
