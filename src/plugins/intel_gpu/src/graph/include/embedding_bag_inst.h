// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/embedding_bag.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {
template <>
struct typed_program_node<embedding_bag> : public typed_program_node_base<embedding_bag> {
    using parent = typed_program_node_base<embedding_bag>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
};

using embedding_bag_node = typed_program_node<embedding_bag>;

template <>
class typed_primitive_inst<embedding_bag> : public typed_primitive_inst_base<embedding_bag> {
    using parent = typed_primitive_inst_base<embedding_bag>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(embedding_bag_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(embedding_bag_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(embedding_bag_node const& node);
    typed_primitive_inst(network& network, embedding_bag_node const& desc);
};

using embedding_bag_inst = typed_primitive_inst<embedding_bag>;
}  // namespace cldnn
