// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/rnn_seq.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {
template <>
struct typed_program_node<rnn_seq> : public typed_program_node_base<rnn_seq> {
    using parent = typed_program_node_base<rnn_seq>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    lstm_weights_order offset_order() const { return get_primitive()->offset_order; }
    float clip() const {
        float clip_val = get_primitive()->clip;
        if (clip_val < 0)
            throw std::range_error("Clip value < 0");
        return clip_val;
    }
    int32_t direction() const { return get_primitive()->direction; }
};

using lstm_seq_node = typed_program_node<rnn_seq>;

template <>
class typed_primitive_inst<rnn_seq> : public typed_primitive_inst_base<rnn_seq> {
    using parent = typed_primitive_inst_base<rnn_seq>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(lstm_seq_node const& node, kernel_impl_params const& impl_param);
    static layout calc_output_layout(lstm_seq_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(lstm_seq_node const& node);

public:
    typed_primitive_inst(network& network, lstm_seq_node const& node);
    lstm_weights_order offset_order() const { return get_typed_desc<rnn_seq>()->offset_order; }
    float clip() const {
        float clip_val = get_typed_desc<rnn_seq>()->clip;
        if (clip_val < 0)
            throw std::range_error("Clip value < 0");
        return clip_val;
    }
    uint32_t direction() const { return get_typed_desc<rnn_seq>()->direction; }
};

using lstm_seq_inst = typed_primitive_inst<rnn_seq>;
}  // namespace cldnn
