// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/ctc_greedy_decoder.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<ctc_greedy_decoder> : public typed_program_node_base<ctc_greedy_decoder> {
    using parent = typed_program_node_base<ctc_greedy_decoder>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    program_node& seq_indicators() const { return get_dependency(1); }

    bool has_second_output() const { return !get_primitive()->second_output.empty(); }
    program_node& second_output() const { return get_dependency(2); }
};

using ctc_greedy_decoder_node = typed_program_node<ctc_greedy_decoder>;

template <>
class typed_primitive_inst<ctc_greedy_decoder> : public typed_primitive_inst_base<ctc_greedy_decoder> {
    using parent = typed_primitive_inst_base<ctc_greedy_decoder>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(ctc_greedy_decoder_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(ctc_greedy_decoder_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(ctc_greedy_decoder_node const& node);

public:
    typed_primitive_inst(network& network, ctc_greedy_decoder_node const& node);
};

using ctc_greedy_decoder_inst = typed_primitive_inst<ctc_greedy_decoder>;

}  // namespace cldnn
