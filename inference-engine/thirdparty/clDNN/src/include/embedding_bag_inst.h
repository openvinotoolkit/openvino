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
#include "api/embedding_bag.hpp"

#include "primitive_inst.h"
#include <string>

namespace cldnn {
template <>
struct typed_program_node<embedding_bag> : public typed_program_node_base<embedding_bag> {
    using parent = typed_program_node_base<embedding_bag>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    size_t inputs_count() const { return get_dependencies().size(); }
};

using embedding_bag_node = typed_program_node<embedding_bag>;

template <>
class typed_primitive_inst<embedding_bag> : public typed_primitive_inst_base<embedding_bag> {
    using parent = typed_primitive_inst_base<embedding_bag>;

public:
    static layout calc_output_layout(embedding_bag_node const& node);
    static std::string to_string(embedding_bag_node const& node);
    typed_primitive_inst(network_impl& network, embedding_bag_node const& desc);
};

using embedding_bag_inst = typed_primitive_inst<embedding_bag>;
}  // namespace cldnn
