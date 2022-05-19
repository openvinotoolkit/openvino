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

layout typed_primitive_inst<dft>::calc_output_layout(const dft_node& node) {
    auto primitive = node.get_primitive();
    auto input_layout = node.input().get_output_layout();

    std::vector<tensor::value_type> dims_converted(primitive->output_shape.begin(), primitive->output_shape.end());
    auto output_format = input_layout.format;

    // Extend shape to 4d by pushing ones before the last dim
    for (auto i = dims_converted.size(); i < 4; ++i) {
        dims_converted.insert(std::prev(dims_converted.end()), 1);
    }

    return {input_layout.data_type, output_format, tensor(output_format, dims_converted)};
}

std::string typed_primitive_inst<dft>::to_string(const dft_node& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    std::ostringstream os;
    node_info->dump(os);
    return os.str();
}

}  // namespace cldnn
