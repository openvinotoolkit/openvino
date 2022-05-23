// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once
#include "intel_gpu/primitives/adaptive_pooling.hpp"
#include "primitive_inst.h"
#include <string>

namespace cldnn {
template <>
struct typed_program_node<adaptive_pooling> : public typed_program_node_base<adaptive_pooling> {
    using parent = typed_program_node_base<adaptive_pooling>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    program_node& output_shape() const { return get_dependency(1); }
    program_node& output_indices() const {
        CLDNN_ERROR_BOOL(id(),
                         "poolingMode != max",
                         get_primitive()->mode != adaptive_pooling_mode::max,
                         "output indices are relevant only for AdaptiveMaxPooling");
        return get_dependency(2);
    }
};

using adaptive_pooling_node = typed_program_node<adaptive_pooling>;

template <>
class typed_primitive_inst<adaptive_pooling> : public typed_primitive_inst_base<adaptive_pooling> {
    using parent = typed_primitive_inst_base<adaptive_pooling>;

public:
    static layout calc_output_layout(const adaptive_pooling_node& node);
    static std::string to_string(const adaptive_pooling_node& node);

    typed_primitive_inst(network& network, const adaptive_pooling_node& node)
            : parent(network, node)
    {}
};

using adaptive_pooling_inst = typed_primitive_inst<adaptive_pooling>;

}  // namespace cldnn
