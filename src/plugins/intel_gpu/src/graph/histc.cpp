// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>
#include <string>

#include "histc_inst.hpp"
#include "json_object.h"
#include "primitive_type_base.h"
#include "to_string_utils.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(histc)

layout histc_inst::calc_output_layout(const histc_node& node, kernel_impl_params const& impl_param) {
    auto primitive = impl_param.typed_desc<histc>();
    return {*primitive->output_data_types[0],
            format::bfyx,
            tensor{1, 1, 1, static_cast<tensor::value_type>(primitive->bins)}};
}

std::string histc_inst::to_string(const histc_node& node) {
    auto primitive = node.get_primitive();
    json_composite histc_info;
    histc_info.add("output_type", dt_to_str(*primitive->output_data_types[0]));
    histc_info.add("bins", primitive->bins);
    histc_info.add("min_val", primitive->min_val);
    histc_info.add("max_val", primitive->max_val);

    auto node_info = node.desc_to_json();
    node_info->add("histc info", histc_info);

    std::ostringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

}  // namespace cldnn
