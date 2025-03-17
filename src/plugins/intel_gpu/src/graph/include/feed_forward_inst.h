// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/feed_forward.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<feed_forward> : public typed_program_node_base<feed_forward> {
    using parent = typed_program_node_base<feed_forward>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using feed_forward_node = typed_program_node<feed_forward>;

template <>
class typed_primitive_inst<feed_forward> : public typed_primitive_inst_base<feed_forward> {
    using parent = typed_primitive_inst_base<feed_forward>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(feed_forward_node const& /*node*/, const kernel_impl_params& impl_params);
    static layout calc_output_layout(feed_forward_node const& node, kernel_impl_params const& impl_params);
    static std::string to_string(feed_forward_node const& node);

    typed_primitive_inst(network& network, feed_forward_node const& node);
};

using feed_forward_inst = typed_primitive_inst<feed_forward>;

}  // namespace cldnn
