// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "api/memory.hpp"
#include "api/primitive.hpp"
#include "api/program.hpp"

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

struct primitive_type {
    virtual ~primitive_type() = default;

    virtual std::shared_ptr<program_node> create_node(program_impl& program,
                                                      const std::shared_ptr<primitive> prim) const = 0;
    virtual std::shared_ptr<primitive_inst> create_instance(network_impl& network,
                                                            const program_node& node) const = 0;
    virtual std::unique_ptr<primitive_impl> choose_impl(engine_impl& engine,
                                                        const program_node& node) const = 0;
    virtual bool does_an_implementation_exist(engine_impl& engine, const program_node& node) const = 0;
    virtual bool does_possible_implementation_exist(engine_impl& engine,
                                                    const program_node& node) const = 0;
    virtual layout calc_output_layout(const program_node& node) const = 0;
    virtual std::string to_string(const program_node& node) const = 0;

    virtual bool is_internal_type() const { return false; }
};
}  // namespace cldnn
