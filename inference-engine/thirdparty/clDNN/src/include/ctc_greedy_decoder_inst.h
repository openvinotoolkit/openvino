/*
// Copyright (c) 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "api/ctc_greedy_decoder.hpp"
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
};

using ctc_greedy_decoder_node = typed_program_node<ctc_greedy_decoder>;

template <>
class typed_primitive_inst<ctc_greedy_decoder> : public typed_primitive_inst_base<ctc_greedy_decoder> {
    using parent = typed_primitive_inst_base<ctc_greedy_decoder>;

public:
    static layout calc_output_layout(ctc_greedy_decoder_node const& node);
    static std::string to_string(ctc_greedy_decoder_node const& node);

public:
    typed_primitive_inst(network_impl& network, ctc_greedy_decoder_node const& node);
};

using ctc_greedy_decoder_inst = typed_primitive_inst<ctc_greedy_decoder>;

}  // namespace cldnn
