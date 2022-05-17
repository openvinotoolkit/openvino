/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_JIT_CONV_KERNEL_ARG_INFO_HPP
#define GPU_JIT_CONV_KERNEL_ARG_INFO_HPP

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/primitive_exec_types.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/jit/conv/ir.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class memory_storage_ptr_t {
public:
    memory_storage_ptr_t(std::unique_ptr<memory_storage_t> &&ptr)
        : unique_ptr_(std::move(ptr)) {}
    memory_storage_ptr_t(const memory_storage_t *ptr) : raw_ptr_(ptr) {}
    memory_storage_ptr_t(const memory_storage_ptr_t &) = delete;

    const memory_storage_t *get() const {
        if (unique_ptr_) return unique_ptr_.get();
        return raw_ptr_;
    }

private:
    std::unique_ptr<memory_storage_t> unique_ptr_; // Owning pointer.
    const memory_storage_t *raw_ptr_ = nullptr; // Non-owning pointer.
};

class memory_storage_wrapper_t {
public:
    memory_storage_wrapper_t() = default;
    memory_storage_wrapper_t(std::unique_ptr<memory_storage_t> &&ptr)
        : ptr_(new memory_storage_ptr_t(std::move(ptr))) {}
    memory_storage_wrapper_t(const memory_storage_t *ptr)
        : ptr_(new memory_storage_ptr_t(ptr)) {}
    memory_storage_wrapper_t(const memory_storage_t &ref)
        : memory_storage_wrapper_t(&ref) {}

    const memory_storage_t *get() const {
        if (!ptr_) return nullptr;
        return ptr_.get()->get();
    }

private:
    std::shared_ptr<memory_storage_ptr_t> ptr_;
};

// Stores kernel arguments. Kernel arguments can be:
// - Internal arguments: only scalar
//   - Examples: common output scales (contain a single value)
// - Resource arguments: stored to a resource storage during primitive creation
//   - Examples: output scales or zero points
// - User arguments: passed by the user at run time
//   - Examples: source, weights, destination
class kernel_arg_info_t {
public:
    void register_internal_arg(const expr_t &var, const expr_t &value) {
        register_arg(var, arg_kind_t::internal, -1, /*is_input=*/true, value);
    }

    void register_resource_arg(const expr_t &var) {
        // TODO: Check key uniqueness.
        register_arg(var, arg_kind_t::resource, nargs(), /*is_input=*/true);
    }

    void register_user_arg(const expr_t &var, int dnnl_arg, bool is_input) {
        register_arg(var, arg_kind_t::user, dnnl_arg, is_input);
    }

    void register_scratchpad_arg(
            const expr_t &var, int key, bool is_input, size_t size) {
        register_arg(
                var, arg_kind_t::scratchpad, key, is_input, expr_t(), size);
    }

    const std::string &arg_name(int idx) const {
        ir_assert(idx >= 0 && idx < nargs());
        return args_[idx].var.as<var_t>().name;
    }

    const expr_t &arg_var(int idx) const {
        ir_assert(idx >= 0 && idx < nargs());
        return args_[idx].var;
    }

    const type_t &arg_type(int idx) const { return arg_var(idx).type(); }

    expr_t find_arg(const std::string &name) const {
        for (int i = 0; i < nargs(); i++) {
            if (arg_name(i) == name) return args_[i].var;
        }
        ir_error_not_expected();
        return expr_t();
    }

    int key(int idx) const {
        ir_assert(idx >= 0 && idx < nargs());
        return args_[idx].key;
    }

    int nargs() const { return int(args_.size()); }

    bool is_resource(int idx) const {
        ir_assert(idx >= 0 && idx < nargs());
        return args_[idx].kind == arg_kind_t::resource;
    }

    bool is_scratchpad(int idx) const {
        ir_assert(idx >= 0 && idx < nargs());
        return args_[idx].kind == arg_kind_t::scratchpad;
    }

    bool is_user(int idx) const {
        ir_assert(idx >= 0 && idx < nargs());
        return args_[idx].kind == arg_kind_t::user;
    }

    bool is_input(int idx) const {
        ir_assert(idx >= 0 && idx < nargs());
        return args_[idx].is_input;
    }

    bool is_output(int idx) const { return !is_input(idx); }

    memory_storage_wrapper_t arg_storage(int idx, const exec_ctx_t &ctx,
            const gpu_primitive_t *primitive) const {
        ir_assert(idx >= 0 && idx < nargs());
        bool is_input = args_[idx].is_input;
        int key = args_[idx].key;
        switch (args_[idx].kind) {
            case arg_kind_t::resource:
#ifdef DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE
                return *(primitive->cached_mapper()
                                 ->template get<gpu_resource_t>(primitive)
                                 ->get_memory_storage(key));
#else
                return *(ctx.get_resource_mapper()
                                 ->get<gpu_resource_t>(primitive)
                                 ->get_memory_storage(key));
#endif
            case arg_kind_t::scratchpad:
                return ctx.get_scratchpad_grantor().get_memory_storage(key);
            case arg_kind_t::user: {
                if (is_input)
                    return ctx.input(args_[idx].key)->memory_storage();
                return ctx.output(args_[idx].key)->memory_storage();
            }
            // No storage for internal arguments.
            case arg_kind_t::internal: return memory_storage_wrapper_t();
            default: ir_error_not_expected();
        }
        return memory_storage_wrapper_t();
    }

    size_t arg_size(int idx, const gpu_primitive_t *primitive) const {
        switch (args_[idx].kind) {
            case arg_kind_t::user: {
                auto *md = primitive->pd()->arg_md(key(idx));
                return memory_desc_wrapper(md).size();
            }
            case arg_kind_t::scratchpad: return args_[idx].scratchpad_size;
            default: ir_error_not_expected();
        }
        return std::numeric_limits<size_t>::max();
    }

    void init_memory_storage_list(std::vector<memory_storage_wrapper_t> &list,
            const exec_ctx_t &ctx, const gpu_primitive_t *primitive) const {
        list = std::vector<memory_storage_wrapper_t>(nargs());
        for (int i = 0; i < nargs(); i++) {
            list[i] = arg_storage(i, ctx, primitive);
        }
    }

    void set_args(compute::kernel_arg_list_t &arg_list,
            const std::vector<memory_storage_wrapper_t> &storage_list) const {
        for (int i = 0; i < nargs(); i++) {
            switch (args_[i].kind) {
                case arg_kind_t::internal: {
                    auto &value = args_[i].value;
                    auto &type = value.type();
                    if (type == type_t::f32()) {
                        arg_list.set(i, to_cpp<float>(value));
                    } else {
                        ir_error_not_expected();
                    }
                    break;
                }
                case arg_kind_t::resource:
                case arg_kind_t::scratchpad:
                case arg_kind_t::user: {
                    arg_list.set(i, *storage_list[i].get());
                    break;
                }
                default: ir_error_not_expected();
            }
        }
    }

private:
    enum class arg_kind_t { internal, resource, scratchpad, user };

    struct arg_t {
        arg_t(const expr_t &var, arg_kind_t kind, int key, bool is_input,
                const expr_t &value, size_t scratchpad_size)
            : var(var)
            , kind(kind)
            , key(key)
            , is_input(is_input)
            , value(value)
            , scratchpad_size(scratchpad_size) {}

        expr_t var;
        arg_kind_t kind;
        int key; // Unique key across arguments with the same kind.
        bool is_input;
        expr_t value; // For internal arguments, must be a constant.
        size_t scratchpad_size; // For scratchpad arguments only.
    };

    void register_arg(const expr_t &var, arg_kind_t kind, int key,
            bool is_input, const expr_t &value = expr_t(),
            size_t scratchpad_size = 0) {
        ir_assert(is_var(var)) << "Expected var, got: " << var;
        args_.emplace_back(var, kind, key, is_input, value, scratchpad_size);
    }

    std::vector<arg_t> args_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
