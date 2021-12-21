// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/roi_pooling.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {
template <>
struct typed_program_node<roi_pooling> : public typed_program_node_base<roi_pooling> {
    using parent = typed_program_node_base<roi_pooling>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    program_node& rois() const { return get_dependency(1); }
    program_node& trans() const { return get_dependency(2); }
};

using roi_pooling_node = typed_program_node<roi_pooling>;

template <>
class typed_primitive_inst<roi_pooling> : public typed_primitive_inst_base<roi_pooling> {
    using parent = typed_primitive_inst_base<roi_pooling>;

public:
    static layout calc_output_layout(roi_pooling_node const& node);
    static std::string to_string(roi_pooling_node const& node);

public:
    using parent::parent;

    memory::ptr rois_memory() const { return dep_memory_ptr(1); }
    memory::ptr trans_memory() const { return dep_memory_ptr(2); }
};

using roi_pooling_inst = typed_primitive_inst<roi_pooling>;

}  // namespace cldnn
