// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/random_uniform.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<random_uniform> : public typed_program_node_base<random_uniform> {
    using parent = typed_program_node_base<random_uniform>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {0}; }
};

using random_uniform_node = typed_program_node<random_uniform>;

template<>
class typed_primitive_inst<random_uniform> : public typed_primitive_inst_base<random_uniform> {
    using parent = typed_primitive_inst_base<random_uniform>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(random_uniform_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(random_uniform_node const &node, kernel_impl_params const& impl_param);

    static std::string to_string(random_uniform_node const &node);

    typed_primitive_inst(network &network, random_uniform_node const &desc);

    // Philox state. It is advanced after every execution so that each inference produces a new sequence.
    uint64_t get_global_seed() const {
        return _global_seed;
    }
    uint64_t get_n_state() const {
        return _n_state;
    }
    uint64_t get_counter_state() const;
    void advance_state(size_t generated_elements);

private:
    uint64_t _global_seed = 0;
    uint64_t _op_seed = 0;
    uint64_t _n_state = 0;
    uint64_t _counter_state = 0;
};

using random_uniform_inst = typed_primitive_inst<random_uniform>;

} // namespace cldnn
