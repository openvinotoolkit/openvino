// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include "json_object.h"
#include "primitive_type_base.h"
#include "recurrent_cache_update_inst.h"
namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(recurrent_cache_update)
layout recurrent_cache_update_inst::calc_output_layout(const recurrent_cache_update_node& n, const kernel_impl_params& p) {
    return calc_output_layouts<ov::PartialShape>(n, p)[0];
}
template <typename ShapeType>
std::vector<layout> recurrent_cache_update_inst::calc_output_layouts(const recurrent_cache_update_node&, const kernel_impl_params& p) {
    const auto d = p.typed_desc<recurrent_cache_update>();
    const bool conv = d->history > 0;
    const auto& data = p.get_input_layout(conv ? 6 : 2);
    auto s = data.get_partial_shape();
    if (conv) {
        if (p.input_layouts.size() == 9)
            s = {s[0], s[1], s[3], s[2]};
        else
            s[3] += d->history;
    }
    const auto& cache = p.get_input_layout(conv ? 0 : 3);
    return {layout(s, data.data_type, format::bfyx), layout(cache.get_partial_shape(), cache.data_type, format::bfyx)};
}
template std::vector<layout> recurrent_cache_update_inst::calc_output_layouts<ov::PartialShape>(const recurrent_cache_update_node&, const kernel_impl_params&);
std::string recurrent_cache_update_inst::to_string(const recurrent_cache_update_node& n) {
    std::stringstream s;
    n.desc_to_json()->dump(s);
    return s.str();
}
recurrent_cache_update_inst::typed_primitive_inst(network& net, const recurrent_cache_update_node& n) : parent(net, n) {}
}  // namespace cldnn
