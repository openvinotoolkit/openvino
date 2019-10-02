// Copyright (c) 2018 Intel Corporation
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

#pragma once
#include "api/pyramid_roi_align.hpp"
#include "primitive_inst.h"

#include <memory>
#include <string>

namespace cldnn {
template <>
struct typed_program_node<pyramid_roi_align> : public typed_program_node_base<pyramid_roi_align> {
    using parent = typed_program_node_base<pyramid_roi_align>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog) : parent(prim, prog) {}

    program_node& input() const { return get_dependency(0); }
    program_node& boxes() const { return get_dependency(0); }
    program_node& image_meta() const { return get_dependency(1); }
    program_node& P2() const { return get_dependency(2); }
    program_node& P3() const { return get_dependency(3); }
    program_node& P4() const { return get_dependency(4); }
    program_node& P5() const { return get_dependency(5); }
    program_node& pool_size() const { return get_dependency(6); }
};

using pyramidROIAlign_node = typed_program_node<pyramid_roi_align>;

template <>
class typed_primitive_inst<pyramid_roi_align> : public typed_primitive_inst_base<pyramid_roi_align> {
    using parent = typed_primitive_inst_base<pyramid_roi_align>;

public:
    static layout calc_output_layout(pyramidROIAlign_node const& node);
    static std::string to_string(pyramidROIAlign_node const& node);
    typed_primitive_inst(network_impl& network, pyramidROIAlign_node const& node);

    memory_impl& boxes() const { return dep_memory(0); }
    memory_impl& image_meta() const { return dep_memory(1); }
    memory_impl& P2() const { return dep_memory(2); }
    memory_impl& P3() const { return dep_memory(3); }
    memory_impl& P4() const { return dep_memory(4); }
    memory_impl& P5() const { return dep_memory(5); }
    memory_impl& pool_size() const { return dep_memory(6); }
};

using pyramid_roi_align_inst = typed_primitive_inst<pyramid_roi_align>;
}  // namespace cldnn