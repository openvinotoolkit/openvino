// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/stateless_kv.hpp"
#include "openvino/core/dimension.hpp"
#include "primitive_inst.h"

#include <optional>
#include <utility>

namespace cldnn {

struct stateless_kv_runtime_state {
    int64_t past_seq_len = 0;
    int64_t present_seq_len = 0;
    int64_t new_token_len = 0;
    int64_t past_front_idx = 0;
    int64_t present_front_idx = 0;
    int64_t sdpa_front_idx = 0;

    bool is_rolling() const {
        return present_front_idx > past_front_idx;
    }

    int64_t get_new_in_present_offset() const {
        return past_seq_len - present_front_idx;
    }

    int64_t get_new_in_sdpa_offset() const {
        return past_seq_len - sdpa_front_idx;
    }

    int64_t get_sdpa_in_present_offset() const {
        return sdpa_front_idx - present_front_idx;
    }

    int64_t get_sdpa_in_past_offset() const {
        return sdpa_front_idx - past_front_idx;
    }

    int64_t get_past_output_len() const {
        return past_seq_len - past_front_idx;
    }

    int64_t get_present_output_len() const {
        return present_seq_len - present_front_idx;
    }

    int64_t get_sdpa_len(int64_t window_size) const {
        return window_size > 0 ? present_seq_len - sdpa_front_idx : present_seq_len;
    }

    layout get_past_to_sdpa_input_layout(const layout& input_layout, int64_t concat_axis) const;

    layout get_sdpa_to_present_input_layout(const layout& input_layout, int64_t concat_axis) const;
};

template <>
struct typed_program_node<stateless_kv> : public typed_program_node_base<stateless_kv> {
private:
    using parent = typed_program_node_base<stateless_kv>;

public:
    using parent::parent;

    program_node& input() const {
        return get_dependency(0);
    }

    std::vector<size_t> get_shape_infer_dependencies() const override {
        return {2};
    }

    std::vector<layout> get_shape_info_input_layouts() const override {
        auto layouts = parent::get_shape_info_input_layouts();
        if (get_primitive()->input.size() != 3)
            return layouts;

        OPENVINO_ASSERT(layouts.size() == 3);
        const auto rank = layouts.front().get_rank();
        const auto axis = get_primitive()->concat_axis;
        OPENVINO_ASSERT(axis >= 0 && axis < static_cast<int64_t>(rank));

        const auto get_dynamic_layout = [&](layout view_layout) {
            auto shape = view_layout.get_partial_shape();
            shape[axis] = ov::Dimension::dynamic();
            view_layout.set_partial_shape(shape);
            view_layout.data_padding._dynamic_dims_mask[axis] = true;
            return view_layout;
        };
        // concat-based always treats input0 as dynamic in case input0 need to be copied
        layouts[0] = get_dynamic_layout(layouts[0]);
        if (get_primitive()->window_size != 0) {
            layouts.push_back(layouts[0]);
            layouts.push_back(is_valid_output_layout(1) ? get_dynamic_layout(get_output_layout(1)) : layouts[0]);
        }
        return layouts;
    }
};

using stateless_kv_node = typed_program_node<stateless_kv>;

template<>
class typed_primitive_inst<stateless_kv> : public typed_primitive_inst_base<stateless_kv> {
    using parent = typed_primitive_inst_base<stateless_kv>;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const stateless_kv_node& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(const stateless_kv_node& node, const kernel_impl_params& impl_param);

    static std::string to_string(const stateless_kv_node& node);

    void update_output_memory() override;

    bool get_is_inplace() const {
        return m_is_inplace;
    }

    const std::optional<stateless_kv_runtime_state>& get_runtime_state() const {
        return m_runtime_state;
    }

    void set_runtime_state(std::optional<stateless_kv_runtime_state> state) {
        m_runtime_state = std::move(state);
    }

    static int64_t get_swa_sequence_length(int64_t abs_seq_len, int64_t cache_capacity, int64_t window_size);
    static std::optional<stateless_kv_runtime_state> compute_runtime_state(const kernel_impl_params& impl_param,
                                                                           const stateless_kv& desc);
    void update_shape_info_tensor(const kernel_impl_params& params) override;

    typed_primitive_inst(network& network, const stateless_kv_node& desc);
    typed_primitive_inst(network& network) : parent(network) {}

    // Kept across executions while present can't be written into the Result buffer in place.
    memory::ptr& private_present() {
        return m_private_present;
    }

private:
    void on_execute() override;
    bool m_is_inplace = false;
    memory::ptr m_private_present;
    std::optional<stateless_kv_runtime_state> m_runtime_state;
};

using stateless_kv_inst = typed_primitive_inst<stateless_kv>;

} // namespace cldnn
