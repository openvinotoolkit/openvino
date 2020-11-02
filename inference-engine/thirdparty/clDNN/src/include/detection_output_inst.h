/*
// Copyright (c) 2016-2020 Intel Corporation
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
#include "api/detection_output.hpp"
#include "primitive_inst.h"
#include "topology_impl.h"
#include <string>

#define PRIOR_BOX_SIZE 4  // Each prior-box consists of [xmin, ymin, xmax, ymax].
#define DETECTION_OUTPUT_ROW_SIZE \
    (3 + PRIOR_BOX_SIZE)  // Each detection consists of [image_id, label, confidence, xmin, ymin, xmax, ymax].

namespace cldnn {

template <>
class typed_program_node<detection_output> : public typed_program_node_base<detection_output> {
    using parent = typed_program_node_base<detection_output>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    program_node& location() const { return get_dependency(0); }
    program_node& confidence() const { return get_dependency(1); }
    program_node& prior_box() const { return get_dependency(2); }
};

using detection_output_node = typed_program_node<detection_output>;

template <>
class typed_primitive_inst<detection_output> : public typed_primitive_inst_base<detection_output> {
    using parent = typed_primitive_inst_base<detection_output>;

public:
    static layout calc_output_layout(detection_output_node const& node);
    static std::string to_string(detection_output_node const& node);

public:
    typed_primitive_inst(network_impl& network, detection_output_node const& node);

    memory_impl& location_memory() const { return dep_memory(0); }
    memory_impl& confidence_memory() const { return dep_memory(1); }
    memory_impl& prior_box_memory() const { return dep_memory(2); }
};

using detection_output_inst = typed_primitive_inst<detection_output>;

}  // namespace cldnn
