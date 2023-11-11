// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/lstm.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {
template <>
struct typed_program_node<lstm_elt> : public typed_program_node_base<lstm_elt> {
    using parent = typed_program_node_base<lstm_elt>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    program_node& cell() const { return get_dependency(1); }
    bool cell_term() const { return !get_primitive()->cell.empty(); }
    lstm_weights_order offset_order() const { return get_primitive()->offset_order; }
    float clip() const {
        float clip_val = get_primitive()->clip;
        if (clip_val < 0)
            throw std::range_error("Clip value < 0");
        return clip_val;
    }
    bool input_forget() const { return get_primitive()->input_forget; }
    int32_t direction() const { return get_primitive()->direction; }
};

using lstm_elt_node = typed_program_node<lstm_elt>;

template <>
class typed_primitive_inst<lstm_elt> : public typed_primitive_inst_base<lstm_elt> {
    using parent = typed_primitive_inst_base<lstm_elt>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(lstm_elt_node const& node, kernel_impl_params const& impl_param);
    static layout calc_output_layout(lstm_elt_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(lstm_elt_node const& node);

public:
    typed_primitive_inst(network& network, lstm_elt_node const& node);

    memory::ptr cell_memory() const { return dep_memory_ptr(1); }
    bool cell_term() const { return !get_typed_desc<lstm_elt>()->cell.empty(); }
    lstm_weights_order offset_order() const { return get_typed_desc<lstm_elt>()->offset_order; }
    float clip() const {
        float clip_val = get_typed_desc<lstm_elt>()->clip;
        if (clip_val < 0)
            throw std::range_error("Clip value < 0");
        return clip_val;
    }
    bool input_forget() const { return get_typed_desc<lstm_elt>()->input_forget; }
    uint32_t direction() const { return get_typed_desc<lstm_elt>()->direction; }
};

using lstm_elt_inst = typed_primitive_inst<lstm_elt>;

}  // namespace cldnn
