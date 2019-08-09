/*
// Copyright (c) 2016 Intel Corporation
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
#include "api/CPP/memory.hpp"
#include "api/CPP/primitive.hpp"
#include "api/CPP/program.hpp"

#include "topology_impl.h"

#include <memory>
#include <string>

namespace cldnn {
struct network_impl;
struct engine_impl;
struct program_node;
struct primitive_impl;
class primitive_inst;
struct program_impl;
}  // namespace cldnn
struct cldnn_primitive_type {
    virtual ~cldnn_primitive_type() = default;

    virtual std::shared_ptr<cldnn::primitive> from_dto(const CLDNN_PRIMITIVE_DESC(primitive) * dto) const = 0;
    virtual std::shared_ptr<cldnn::program_node> create_node(cldnn::program_impl& program,
                                                             const std::shared_ptr<cldnn::primitive> prim) const = 0;
    virtual std::shared_ptr<cldnn::primitive_inst> create_instance(cldnn::network_impl& network,
                                                                   const cldnn::program_node& node) const = 0;
    virtual std::unique_ptr<cldnn::primitive_impl> choose_impl(cldnn::engine_impl& engine,
                                                               const cldnn::program_node& node) const = 0;
    virtual bool does_an_implementation_exist(cldnn::engine_impl& engine, const cldnn::program_node& node) const = 0;
    virtual bool does_possible_implementation_exist(cldnn::engine_impl& engine,
                                                    const cldnn::program_node& node) const = 0;
    virtual cldnn::layout calc_output_layout(const cldnn::program_node& node) const = 0;
    virtual std::string to_string(const cldnn::program_node& node) const = 0;

    virtual bool is_internal_type() const { return false; }
};
