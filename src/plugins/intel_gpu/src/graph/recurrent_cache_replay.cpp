// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "json_object.h"
#include "primitive_type_base.h"
#include "recurrent_cache_replay_inst.h"
namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(recurrent_cache_replay)
layout recurrent_cache_replay_inst::calc_output_layout(const recurrent_cache_replay_node& n,
                                                       const kernel_impl_params& p) {
    return calc_output_layouts<ov::PartialShape>(n, p)[0];
}
template <typename ShapeType>
std::vector<layout> recurrent_cache_replay_inst::calc_output_layouts(const recurrent_cache_replay_node&,
                                                                     const kernel_impl_params& p) {
    const auto d = p.typed_desc<recurrent_cache_replay>();
    const bool conv = d->history > 0;
    const auto& data = p.get_input_layout(2);
    auto s = data.get_partial_shape();
    return {layout(s, p.get_input_layout(conv ? 2 : 0).data_type, format::bfyx)};
}
template std::vector<layout>
recurrent_cache_replay_inst::calc_output_layouts<ov::PartialShape>(const recurrent_cache_replay_node&,
                                                                   const kernel_impl_params&);
std::string recurrent_cache_replay_inst::to_string(const recurrent_cache_replay_node& n) {
    std::stringstream s;
    n.desc_to_json()->dump(s);
    return s.str();
}
recurrent_cache_replay_inst::typed_primitive_inst(network& net, const recurrent_cache_replay_node& n)
    : parent(net, n) {}

} // namespace cldnn
