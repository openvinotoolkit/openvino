// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/gemm.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {
template <>
struct typed_program_node<gemm> : public typed_program_node_base<gemm> {
    using parent = typed_program_node_base<gemm>;

public:
    using parent::parent;

    program_node& input(size_t idx = 0) const { return get_dependency(idx); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using gemm_node = typed_program_node<gemm>;

template <>
class typed_primitive_inst<gemm> : public typed_primitive_inst_base<gemm> {
    using parent = typed_primitive_inst_base<gemm>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(gemm_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(gemm_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(gemm_node const& node);

    static std::vector<layout> transform_input_layouts(const std::shared_ptr<const gemm> primitive,
                                                       const std::vector<layout>& input_layouts,
                                                       const bool allow_new_shape_infer);
    static layout transform_output_layout(const std::shared_ptr<const gemm> primitive, const std::vector<layout>& input_layouts, const layout& output_layout);

    static bool is_fusable_permute_input_order_onednn(const std::vector<size_t>& permute_order, format& fmt) {
        const std::vector<format> gemm_in_format_white_list = {format::bfyx,
                                                               format::bfxy,
                                                               format::fyxb,
                                                               format::byfx,
                                                               format::bxfy,
                                                               format::fybx,
                                                               format::ybfx,
                                                               format::xbfy};
        auto target_permute_order = permute_order;
        for (size_t i = 0; i < permute_order.size(); ++i) {
            target_permute_order[permute_order[i]] = i;
        }

        for (const auto& cand_format : gemm_in_format_white_list) {
            const auto cand_format_order = format::traits(static_cast<format::type>(cand_format))._order;
            if (cand_format_order == target_permute_order) {
                fmt = cand_format;
                return true;
            }
        }
        return false;
    }

    static bool is_fusable_permute_output_order_onednn(const std::vector<size_t>& target_order, format& fmt) {
        const std::vector<format> gemm_out_format_white_list = {format::bfyx,
                                                                format::bfxy,
                                                                format::fyxb,
                                                                format::fybx,
                                                                format::byfx,
                                                                format::ybfx};

        for (const auto& cand_format : gemm_out_format_white_list) {
            const auto cand_format_order = format::traits(static_cast<format::type>(cand_format))._order;
            if (cand_format_order == target_order) {
                fmt = cand_format;
                return true;
            }
        }
        return false;
    }


    typed_primitive_inst(network& network, gemm_node const& node);
};

using gemm_inst = typed_primitive_inst<gemm>;

}  // namespace cldnn
