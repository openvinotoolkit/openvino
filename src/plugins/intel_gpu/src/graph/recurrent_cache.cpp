// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "json_object.h"
#include "primitive_type_base.h"
#include "recurrent_cache_inst.h"
namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(recurrent_cache)
layout recurrent_cache_inst::calc_output_layout(const recurrent_cache_node& node, const kernel_impl_params& params) {
    return calc_output_layouts<ov::PartialShape>(node, params)[0];
}
template <typename ShapeType>
std::vector<layout> recurrent_cache_inst::calc_output_layouts(const recurrent_cache_node&, const kernel_impl_params& params) {
    const auto desc = params.typed_desc<recurrent_cache>();
    const auto cache = params.get_input_layout(0);
    if (desc->history < 0)
        return {layout(cache.get_partial_shape(), cache.data_type, format::bfyx)};
    ov::PartialShape active;
    auto type = cache.data_type;
    if (desc->history == 0) {
        active = {1, 1, params.get_input_layout(1).get_partial_shape()[3], cache.get_partial_shape()[3]};
    } else {
        active = params.get_input_layout(6).get_partial_shape();
        active[3] += desc->history;
        type = params.get_input_layout(6).data_type;
    }
    return {layout(active, type, format::bfyx), layout(cache.get_partial_shape(), cache.data_type, format::bfyx)};
}
template std::vector<layout> recurrent_cache_inst::calc_output_layouts<ov::PartialShape>(const recurrent_cache_node&, const kernel_impl_params&);
std::string recurrent_cache_inst::to_string(const recurrent_cache_node& node) {
    std::stringstream ss;
    node.desc_to_json()->dump(ss);
    return ss.str();
}
recurrent_cache_inst::typed_primitive_inst(network& network, const recurrent_cache_node& node) : parent(network, node) {}
}  // namespace cldnn
