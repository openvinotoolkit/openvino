// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/rnn.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {
template <>
struct typed_program_node<gru_seq> : public typed_program_node_base<gru_seq> {
    using parent = typed_program_node_base<gru_seq>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    lstm_weights_order offset_order() const { return get_primitive()->offset_order; }
    float clip() const {
        float clip_val = get_primitive()->clip;
        OPENVINO_ASSERT(clip_val >= 0, "Clip value < 0");
        return clip_val;
    }
    ov::op::RecurrentSequenceDirection direction() const { return get_primitive()->direction; }
    std::vector<activation_func> activations() const {
        return get_primitive()->activations;
    }
    bool permute_inserted = false;
};

using gru_seq_node = typed_program_node<gru_seq>;

template <>
class typed_primitive_inst<gru_seq> : public typed_primitive_inst_base<gru_seq> {
    using parent = typed_primitive_inst_base<gru_seq>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(gru_seq_node const& node, kernel_impl_params const& impl_param);
    static layout calc_output_layout(gru_seq_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(gru_seq_node const& node);

public:
    typed_primitive_inst(network& network, gru_seq_node const& node);
    lstm_weights_order offset_order() const { return get_typed_desc<gru_seq>()->offset_order; }
    float clip() const {
        float clip_val = get_typed_desc<gru_seq>()->clip;
        if (clip_val < 0)
            throw std::range_error("Clip value < 0");
        return clip_val;
    }
    uint32_t direction() const { return get_typed_desc<gru_seq>()->direction == ov::op::RecurrentSequenceDirection::FORWARD ? 0 : 1; }
    bool has_cell() const { return !get_typed_desc<gru_seq>()->initial_cell_state.pid.empty(); }
};

using gru_seq_inst = typed_primitive_inst<gru_seq>;
}  // namespace cldnn
