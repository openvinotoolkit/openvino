// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lrn_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id lrn::type_id() {
    static primitive_type_base<lrn> instance;
    return &instance;
}

layout lrn_inst::calc_output_layout(lrn_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for lrn_node!");
    auto input_layout = node.input().get_output_layout();
    auto output_type = input_layout.data_type;

    if (node.has_fused_primitives()) {
        output_type = node.get_fused_output_layout().data_type;
    }

    auto result = node.input().get_non_padded_output_layout();
    result.data_type = output_type;

    return result;
}

std::string lrn_inst::to_string(lrn_node const& node) {
    auto node_info = node.desc_to_json();
    auto desc = node.get_primitive();
    auto norm_size = desc->size;
    auto k = desc->k;
    auto alpha = desc->alpha;
    auto beta = desc->beta;
    auto norm_region = desc->norm_region == lrn_norm_region::lrn_norm_region_across_channel
                           ? "across channel"
                           : "within channel";
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite lrn_info;
    lrn_info.add("input id", input.id());
    lrn_info.add("k", k);
    lrn_info.add("alpha", alpha);
    lrn_info.add("beta", beta);
    lrn_info.add("size of normalization", norm_size);
    lrn_info.add("normalization region", norm_region);

    node_info->add("lrn info", lrn_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

lrn_inst::typed_primitive_inst(network& network, lrn_node const& desc) : parent(network, desc) {
    CLDNN_ERROR_LESS_OR_EQUAL_THAN(desc.id(),
                                   "LRN argument size",
                                   argument.size,
                                   "value",
                                   static_cast<uint32_t>(0),
                                   "LRN size must be greater than 0!");
}
}  // namespace cldnn
