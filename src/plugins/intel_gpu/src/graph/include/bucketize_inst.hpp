// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/bucketize.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<bucketize> : typed_program_node_base<bucketize> {
    using parent = typed_program_node_base<bucketize>;
    using parent::parent;

    program_node& input() const {
        return get_dependency(0);
    }
    program_node& buckets() const {
        return get_dependency(1);
    }
};

using bucketize_node = typed_program_node<bucketize>;

template <>
class typed_primitive_inst<bucketize> : public typed_primitive_inst_base<bucketize> {
public:
    using parent = typed_primitive_inst_base<bucketize>;
    using parent::parent;

    static layout calc_output_layout(const bucketize_node& node);
    static std::string to_string(const bucketize_node& node);
};

using bucketize_inst = typed_primitive_inst<bucketize>;

}  // namespace cldnn
