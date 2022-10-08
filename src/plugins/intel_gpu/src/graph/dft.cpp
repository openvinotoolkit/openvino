// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <dft_inst.h>
#include <primitive_type_base.h>

#include "json_object.h"

namespace cldnn {

primitive_type_id dft::type_id() {
    static primitive_type_base<dft> instance;
    return &instance;
}

layout dft_inst::calc_output_layout(const dft_node& node, const kernel_impl_params& impl_param) {
    const auto primitive = impl_param.typed_desc<dft>();
    const auto input_layout = impl_param.get_input_layout();

    std::vector<tensor::value_type> dims_converted(primitive->output_shape.begin(), primitive->output_shape.end());

    // Extend output layout for IRDFT case to make output rank match input rank
    if (primitive->direction == dft_direction::inverse && primitive->mode == dft_mode::real) {
        dims_converted.push_back(1);
    }

    // Extend shape to 4d by pushing ones before the last dim
    for (auto i = dims_converted.size(); i < 4; ++i) {
        dims_converted.insert(std::prev(dims_converted.end()), 1);
    }

    const auto output_format = format::adjust_to_rank(input_layout.format, dims_converted.size());
    return {input_layout.data_type, output_format, tensor(output_format, dims_converted)};
}

std::string dft_inst::to_string(const dft_node& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    std::ostringstream os;
    node_info->dump(os);
    return os.str();
}

}  // namespace cldnn
