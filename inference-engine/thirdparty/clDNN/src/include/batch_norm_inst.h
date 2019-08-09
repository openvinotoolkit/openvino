/*
// Copyright (c) 2016 Intel Corporation
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
#include "api/CPP/batch_norm.hpp"
#include "primitive_inst.h"
#include "mutable_data_inst.h"
#include <string>

namespace cldnn {

template <>
struct typed_program_node<batch_norm> : public typed_program_node_base<batch_norm> {
    using parent = typed_program_node_base<batch_norm>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    program_node& mean() const { return get_dependency(1); }
    program_node& variance() const { return get_dependency(2); }
    program_node& scale() const {
        if (get_dependencies().size() >= 5)
            return get_dependency(3);
        else
            return get_dependency(1);
    }
    program_node& shift() const {
        if (get_dependencies().size() >= 5)
            return get_dependency(4);
        else
            return get_dependency(2);
    }
    program_node& inv_variance() const {
        if (get_dependencies().size() == 2)
            return get_dependency(1);
        else if (get_dependencies().size() == 6)
            return get_dependency(5);
        else
            return get_dependency(3);
    }
    bool variance_term() const { return !get_primitive()->variance.empty(); }
    bool use_global_stats() const { return !get_primitive()->mean.empty() && !get_primitive()->variance.empty(); }
    bool use_scale_shift() const { return !get_primitive()->scale.empty() && !get_primitive()->shift.empty(); }
    bool forwad_pass() const { return !get_primitive()->inv_variance.empty(); }
    bool calc_mean_var() const {
        return (use_global_stats() && mean().is_type<mutable_data>() && variance().is_type<mutable_data>());
    }
};

using batch_norm_node = typed_program_node<batch_norm>;

template <>
class typed_primitive_inst<batch_norm> : public typed_primitive_inst_base<batch_norm> {
    using parent = typed_primitive_inst_base<batch_norm>;

public:
    static layout calc_output_layout(batch_norm_node const& node);
    static std::string to_string(batch_norm_node const& node);

public:
    typed_primitive_inst(network_impl& network, batch_norm_node const& node);

    memory_impl& mean_memory() const { return dep_memory(1); }
    memory_impl& variance_memory() const { return dep_memory(2); }
    memory_impl& scale_memory() const {
        if (dependencies().size() >= 5)
            return dep_memory(3);
        else
            return dep_memory(1);
    }
    memory_impl& shift_memory() const {
        if (dependencies().size() >= 5)
            return dep_memory(4);
        else
            return dep_memory(2);
    }
    memory_impl& inv_variance_memory() const {
        if (dependencies().size() == 2)
            return dep_memory(1);
        else if (dependencies().size() == 6)
            return dep_memory(5);
        else
            return dep_memory(3);
    }
    bool use_global_stats() const { return !argument.mean.empty() && !argument.variance.empty(); }
    bool use_scale_shift() const { return !argument.scale.empty() && !argument.scale.empty(); }
    bool forwad_pass() const { return !argument.inv_variance.empty(); }
    bool calc_mean_var() const { return node.calc_mean_var(); }
};

using batch_norm_inst = typed_primitive_inst<batch_norm>;

}  // namespace cldnn
