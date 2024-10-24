// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/lstm_cell.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {
template <>
struct typed_program_node<lstm_cell> : public typed_program_node_base<lstm_cell> {
    using parent = typed_program_node_base<lstm_cell>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    lstm_weights_order offset_order() const { return get_primitive()->offset_order; }
    float clip() const {
        float clip_val = get_primitive()->clip;
        OPENVINO_ASSERT(clip_val >= 0, "Clip value < 0");
        return clip_val;
    }
    int32_t direction() const { return get_primitive()->direction; }
};

using lstm_cell_node = typed_program_node<lstm_cell>;

template <>
class typed_primitive_inst<lstm_cell> : public typed_primitive_inst_base<lstm_cell> {
    using parent = typed_primitive_inst_base<lstm_cell>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(lstm_cell_node const& node, kernel_impl_params const& impl_param);
    static layout calc_output_layout(lstm_cell_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(lstm_cell_node const& node);

public:
    typed_primitive_inst(network& network, lstm_cell_node const& node);
    lstm_weights_order offset_order() const { return get_typed_desc<lstm_cell>()->offset_order; }
    float clip() const {
        float clip_val = get_typed_desc<lstm_cell>()->clip;
        OPENVINO_ASSERT(clip_val >= 0, "Clip value < 0");
        return clip_val;
    }
    uint32_t direction() const { return get_typed_desc<lstm_cell>()->direction; }
    bool has_cell() const { return !get_typed_desc<lstm_cell>()->initial_cell_state.pid.empty(); }
};

using lstm_cell_inst = typed_primitive_inst<lstm_cell>;
}  // namespace cldnn
