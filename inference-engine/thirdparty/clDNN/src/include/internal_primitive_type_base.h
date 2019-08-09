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
#pragma once

#include "meta_utils.h"
#include "primitive_type.h"
#include "internal_primitive.h"
#include "program_node.h"
#include "primitive_inst.h"
#include <string>
#include <memory>

namespace cldnn {

template <class PType>
struct internal_primitive_type_base : public ::cldnn_primitive_type {
    static_assert(meta::is_internal_primitive<PType>::value,
                  "Primitive type passed to internal_primitive_type_base should derive from internal_primitive");

    [[noreturn]] std::shared_ptr<primitive> from_dto(const CLDNN_PRIMITIVE_DESC(primitive) *) const override {
        throw std::runtime_error(
            "Trying to create an internal primitive from dto - internal primitives are intransferable by design");
    }

    [[noreturn]] std::shared_ptr<cldnn::program_node> create_node(program_impl&,
                                                                  const std::shared_ptr<primitive>) const override {
        throw std::runtime_error(
            "Trying to create generic program_node for an internal primitive - internal primitives' nodes should be "
            "created manually");
    }

    std::shared_ptr<cldnn::primitive_inst> create_instance(network_impl& network,
                                                           const cldnn::program_node& node) const override {
        if (node.type() != this)
            throw std::invalid_argument("internal_primitive_type_base::create_instance: primitive type mismatch");

        return std::make_shared<typed_primitive_inst<PType>>(network, node);
    }

    [[noreturn]] std::unique_ptr<primitive_impl> choose_impl(cldnn::engine_impl&,
                                                             const cldnn::program_node&) const override {
        throw std::runtime_error(
            "primitive_type_id::choose_impl called for internal primitive - internal primitives should have manually "
            "attached executable");
    }

    [[noreturn]] cldnn::layout calc_output_layout(const cldnn::program_node&) const override {
        throw std::runtime_error(
            "primitive_type_id::calc_output_layout called for internal primitive - internal primitives should have "
            "output layouts precalculated");
    }

    std::string to_string(const cldnn::program_node& node) const override {
        if (node.type() != this)
            throw std::invalid_argument("primitive_type_base::to_string: primitive type mismatch");

        return typed_primitive_inst<PType>::to_string(node);
    }

    bool is_internal_type() const override { return true; }
};

#define CLDNN_DEFINE_INTERNAL_PRIM(PType)                        \
    struct PType : public internal_primitive {                   \
        static primitive_type_id type_id() {                     \
            static internal_primitive_type_base<PType> instance; \
            return &instance;                                    \
        }                                                        \
    };                                                           \
    using PType##_node = typed_program_node<PType>;

}  // namespace cldnn
