// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "meta_utils.h"
#include "primitive_type.h"
#include "program_node.h"
#include "primitive_inst.h"
#include "network_impl.h"
#include "engine_impl.h"
#include <memory>
#include <string>

namespace cldnn {
template <class PType>
struct primitive_type_base : primitive_type {
    static_assert(meta::is_api_primitive<PType>::value,
                  "Primitive type passed to primitive_type_base should derive from cldnn::primitive");

    std::shared_ptr<cldnn::program_node> create_node(program_impl& program,
                                                     const std::shared_ptr<primitive> prim) const override {
        if (prim->type != this)
            throw std::invalid_argument("primitive_type_base::create_node: primitive type mismatch");

        return std::make_shared<typed_program_node<PType>>(std::static_pointer_cast<PType>(prim), program);
    }

    std::shared_ptr<cldnn::primitive_inst> create_instance(network_impl& network,
                                                           const cldnn::program_node& node) const override {
        if (node.type() != this)
            throw std::invalid_argument("primitive_type_base::create_instance: primitive type mismatch");

        return std::make_shared<typed_primitive_inst<PType>>(network, node);
    }

    std::unique_ptr<primitive_impl> choose_impl(engine_impl& engine, const cldnn::program_node& node) const override {
        if (node.type() != this)
            throw std::invalid_argument("primitive_type_base::choose_impl: primitive type mismatch");

        return engine.create_primitive_impl(node.as<PType>());
    }

    bool does_an_implementation_exist(engine_impl& engine, const cldnn::program_node& node) const override {
        if (node.type() != this)
            throw std::invalid_argument("primitive_type_base::choose_impl: primitive type mismatch");
        return engine.does_an_implementation_exist(node.as<PType>());
    }

    bool does_possible_implementation_exist(engine_impl& engine, const cldnn::program_node& node) const override {
        if (node.type() != this)
            throw std::invalid_argument("primitive_type_base::choose_impl: primitive type mismatch");
        return engine.does_possible_implementation_exist(node.as<PType>());
    }

    cldnn::layout calc_output_layout(const cldnn::program_node& node) const override {
        if (node.type() != this)
            throw std::invalid_argument("primitive_type_base::calc_output_layout: primitive type mismatch");

        return typed_primitive_inst<PType>::calc_output_layout(node);
    }

    std::string to_string(const cldnn::program_node& node) const override {
        if (node.type() != this)
            throw std::invalid_argument("primitive_type_base::to_string: primitive type mismatch");

        return typed_primitive_inst<PType>::to_string(node);
    }
};

}  // namespace cldnn
