/*
// Copyright (c) 2017 Intel Corporation
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
#include "api/CPP/roi_pooling.hpp"
#include "primitive_inst.h"

namespace cldnn
{
template <>
struct typed_program_node<roi_pooling> : public typed_program_node_base<roi_pooling>
{
    using parent = typed_program_node_base<roi_pooling>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    program_node& rois() const { return get_dependency(1); }
};

using roi_pooling_node = typed_program_node<roi_pooling>;

template <>
class typed_primitive_inst<roi_pooling> : public typed_primitive_inst_base<roi_pooling>
{
    using parent = typed_primitive_inst_base<roi_pooling>;

public:
    static layout calc_output_layout(roi_pooling_node const& node);
    static std::string to_string(roi_pooling_node const& node);

public:    
    using parent::parent;

    memory_impl& rois_memory() const { return dep_memory(1); }
};

using roi_pooling_inst = typed_primitive_inst<roi_pooling>;

}
