// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/kv_cache.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/partial_shape.hpp"
#include "primitive_inst.h"
#include "variable.hpp"

namespace cldnn {

template <>
struct typed_program_node<kv_cache> : public typed_program_node_base<kv_cache> {
private:
    using parent = typed_program_node_base<kv_cache>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }

    std::vector<layout> get_shape_info_input_layouts() const override {
        std::vector<layout> res;
        for (size_t i = 0; i < get_dependencies().size(); i++) {
            const auto& d = get_dependency_with_port(i);
            res.push_back(d.first->get_output_layout(false, d.second));
        }

        if (get_primitive()->indirect) { // insert an additional input with beam table past layout
            res.push_back(layout(ov::PartialShape::dynamic(4), data_types::i32, format::bfyx));
        }

        return res;
    }
};

using kv_cache_node = typed_program_node<kv_cache>;

template<>
class typed_primitive_inst<kv_cache> : public typed_primitive_inst_base<kv_cache>, public memory_state::variable {
    using parent = typed_primitive_inst_base<kv_cache>;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(kv_cache_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(const kv_cache_node& node, kernel_impl_params const& impl_param);

    static std::string to_string(const kv_cache_node& node);

    int32_t get_prealloc_iter_num() override;

    static void update_pad(layout& l, int64_t pad, int64_t sequence_axis) {
        l.data_padding._upper_size[sequence_axis] = static_cast<int32_t>(pad);
    }

    static int64_t get_sequence_axis(int64_t sequence_axis, size_t past_layout_rank) {
        return sequence_axis >= 0 ? sequence_axis : past_layout_rank + sequence_axis;
    }

    static int64_t get_scale_zp_sequence_axis() {
        // The order of scales and zero points is fixed, so use constant axis
        const auto scale_zp_concat_axis = 2;
        return scale_zp_concat_axis;
    }

    static int64_t get_max_pad(const layout& target_layout, size_t buffer_size, int64_t sequence_axis, std::string target_name = "") {
        if (buffer_size == 0)
            return 0;
        const size_t total_elements = target_layout.count();
        const int64_t concat_axis_size = target_layout.get_shape()[sequence_axis];
        const int64_t sequence_element_size = total_elements / concat_axis_size;
        const int64_t max_sequence_elements = buffer_size / sequence_element_size;
        auto max_pad = std::max<int64_t>(max_sequence_elements - concat_axis_size, 0);
        auto target_layout_name = (target_name != "") ? target_name : "target_layout";
        GPU_DEBUG_TRACE_DETAIL << "[get_max_pad] " << target_name  << " : " << target_layout.to_string() << std::endl;
        GPU_DEBUG_TRACE_DETAIL << "[get_max_pad] buffer size " << buffer_size << std::endl;
        GPU_DEBUG_TRACE_DETAIL << "[get_max_pad] total_elements " << total_elements << std::endl;
        GPU_DEBUG_TRACE_DETAIL << "[get_max_pad] concat_axis_size = " << concat_axis_size << std::endl;
        GPU_DEBUG_TRACE_DETAIL << "[get_max_pad] sequence_element_size = " << sequence_element_size << std::endl;
        GPU_DEBUG_TRACE_DETAIL << "[get_max_pad] max_sequence_elements = " << max_sequence_elements << std::endl;
        GPU_DEBUG_TRACE_DETAIL << "[get_max_pad] max_pad (max_sequence_elements - concat_axis_size) = " << max_pad << std::endl;
        return max_pad;
    }
    void update_shape_info_tensor(const kernel_impl_params& params) override;

    typed_primitive_inst(network& network, const kv_cache_node& desc);
    typed_primitive_inst(network& network) : parent(network), memory_state::variable("") {}

private:
    size_t kv_cache_id = 0;
};

using kv_cache_inst = typed_primitive_inst<kv_cache>;

} // namespace cldnn
