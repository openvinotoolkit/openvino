// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "intel_gpu/runtime/compounds.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/optionals.hpp"

#include <algorithm>
#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <utility>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{

/// @addtogroup cpp_topology Network Topology
/// @{

/// @brief Globally unique primitive's type id
using primitive_type_id = struct primitive_type *;

/// @brief Unique @p id of a primitive within a topology.
using primitive_id = std::string;

struct primitive_info;

struct input_info {
    input_info() : pid(""), idx(0) {}
    input_info(primitive_id pid) : pid(pid), idx(0) {}
    input_info(primitive_id pid, int idx) : pid(pid), idx(idx) {}

    primitive_id pid;
    int32_t idx;
    struct cmp {
        bool operator() (const input_info a, const input_info b) {
            if (a.pid < b.pid) {
                return true;
            } else if (a.pid == b.pid) {
                return a.idx < b.idx;
            } else {
                return false;
            }
        }
    };
};

/// @brief Base class of network primitive description.
struct primitive {
public:
    /// @brief Initialize fields common for all primitives.
    primitive(const primitive_type_id& type,
              const primitive_id& id,
              const std::vector<primitive_id>& input,
              const padding& output_padding = padding(),
              const optional_data_type output_data_type = optional_data_type(),
              const std::vector<input_info>& input_new = {},
              const size_t num_outputs = 1)
        : type(type),
          id(id),
          output_padding(output_padding),
          output_data_type(output_data_type),
          input(input),
          input_new(input_new),
          num_outputs(num_outputs) {}

    virtual ~primitive() = default;

    /// @brief Returns references to all primitive ids on which this primitive depends - inputs, weights, biases, etc.
    std::vector<std::reference_wrapper<primitive_id>> dependencies() {
        std::vector<std::reference_wrapper<primitive_id>> result;
        auto&& deps = get_dependencies();

        result.reserve(input.size() + deps.size());
        for (auto& pid : input) result.push_back(std::ref(pid));
        for (auto& pid : deps) result.push_back(std::ref(const_cast<primitive_id&>(pid.get())));

        return result;
    }

    /// @brief Returns copy of all primitive ids on which this primitive depends - inputs, weights, biases, etc.
    std::vector<primitive_id> dependencies() const {
        auto result = input;
        auto deps = get_dependencies();
        result.insert(result.end(), deps.begin(), deps.end());
        return result;
    }

    std::vector<std::reference_wrapper<input_info>> dependencies_new() {
        std::vector<std::reference_wrapper<input_info>> result;
        auto&& deps = get_dependencies_new();
        result.reserve(input_new.size() + deps.size());
        for (auto& i : input_new) result.push_back(std::ref(i));
        for (auto& dep : deps) result.push_back({std::ref(const_cast<input_info&>(dep.get()))});

        return result;
    }

    std::vector<input_info> dependencies_new() const {
        auto result = input_new;
        auto deps = get_dependencies_new();
        result.insert(result.end(), deps.begin(), deps.end());
        return result;
    }

    virtual primitive_id type_string() const = 0;

    /// @brief Implicit conversion to primiitive id.
    operator primitive_id() const { return id; }

    /// @brief Primitive's type id.
    const primitive_type_id type;

    /// @brief Primitive's id.
    const primitive_id id;

    /// @brief Name of original ov operation.
    std::string origin_op_name;

    /// @brief Type name of original ov operation.
    std::string origin_op_type_name;

    /// @brief Requested output padding.
    padding output_padding;

    /// @brief Requested output precision, if any.
    optional_data_type output_data_type;

    size_t input_size() const { return input.size(); }

    size_t output_size() const { return num_outputs; }

    using primitive_id_arr = std::vector<primitive_id>;

    /// @brief List of ids of input primitives.
    primitive_id_arr input;

    using input_info_arr = std::vector<input_info>;

    /// @brief List of input info containing id and output index of input primitive.
    input_info_arr input_new;

    size_t num_outputs;

protected:
    virtual std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const { return {}; }
    virtual std::vector<std::reference_wrapper<const input_info>> get_dependencies_new() const { return {}; }
    class condition;
    friend struct primitive_info;
};

/// @brief base class for all primitives implementations.
template <class PType>
class primitive_base : public primitive {
protected:
    explicit primitive_base(const primitive_id& id,
                            const std::vector<primitive_id>& input,
                            const padding& output_padding = padding(),
                            optional_data_type output_data_type = optional_data_type(),
                            const std::vector<input_info>& input_new = {},
                            const size_t num_outputs = 1)
        : primitive(PType::type_id(), id, input, output_padding, output_data_type, input_new, num_outputs) {}
};

struct primitive_info {
    primitive_info(const primitive_id& original_id,
                   const std::string& type_id,
                   const std::vector<primitive_id>& dependencies,
                   const std::vector<primitive_id>& users,
                   const std::vector<primitive_id>& fused_ids,
                   const layout& output_layout,
                   const std::string& layout_str,
                   const std::string& kernel_id,
                   const data_types& runtime_precision,
                   bool is_cpu,
                   int exec_id)
        : original_id(original_id),
          type_id(type_id),
          c_dependencies(dependencies),
          c_users(users),
          c_fused_ids(fused_ids),
          output_layout(output_layout),
          layout_str(layout_str),
          kernel_id(kernel_id),
          runtime_precision(runtime_precision),
          is_cpu(is_cpu),
          exec_id(exec_id) {}

    primitive_id original_id;
    std::string type_id;
    primitive::primitive_id_arr c_dependencies;
    primitive::primitive_id_arr c_users;
    primitive::primitive_id_arr c_fused_ids;
    layout output_layout;
    std::string layout_str;
    std::string kernel_id;
    data_types runtime_precision;
    bool is_cpu;
    int exec_id;
};

#define CLDNN_DEFINE_TYPE_ID(PType)     \
    static primitive_type_id type_id();

#define CLDNN_DEFINE_TYPE_STRING(PType)                 \
    primitive_id type_string() const override {         \
        static constexpr const char* type_str = #PType; \
        return std::string(type_str);                   \
    }

#define CLDNN_DECLARE_PRIMITIVE(PType)       \
    CLDNN_DEFINE_TYPE_ID(PType)              \
    CLDNN_DEFINE_TYPE_STRING(PType)

/// @}
/// @}
}  // namespace cldnn
