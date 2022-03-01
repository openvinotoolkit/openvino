// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>

#include "intel_gpu/primitives/bucketize.hpp"
#include "primitive_inst.h"

namespace cldnn {
template <>
struct typed_program_node<bucketize> : public typed_program_node_base<bucketize> {
    using parent = typed_program_node_base<bucketize>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const {
        return get_dependency(index);
    }
};

using bucketize_node = typed_program_node<bucketize>;

template <>
class typed_primitive_inst<bucketize> : public typed_primitive_inst_base<bucketize> {
    using parent = typed_primitive_inst_base<bucketize>;

public:
    static layout calc_output_layout(bucketize_node const& node);
    static std::string to_string(bucketize_node const& node);

public:
    typed_primitive_inst(network& network, bucketize_node const& desc);
};

using bucketize_inst = typed_primitive_inst<bucketize>;
}  // namespace cldnn
