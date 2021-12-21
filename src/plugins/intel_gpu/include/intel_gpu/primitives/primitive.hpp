// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "intel_gpu/runtime/compounds.hpp"
#include "intel_gpu/runtime/layout.hpp"

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
              const std::vector<input_info>& input,
              const primitive_id& ext_prim_id = "",
              const std::vector<padding>& output_paddings = {padding()},
              const std::vector<optional_data_type> output_data_types = {optional_data_type()}, // TODO: change for multiple output
              const int num_outputs = 1)
        : type(type),
          id(id),
          ext_prim_id(ext_prim_id),
          output_paddings(output_paddings),
          output_data_types(output_data_types),
          input(input),
          num_outputs(num_outputs) {}

    virtual ~primitive() = default;

    /// @brief Returns references to all primitive ids on which this primitive depends - inputs, weights, biases, etc.
    std::vector<input_info> dependencies() {
        std::vector<input_info> result;
        auto&& deps = get_dependencies();
        result.reserve(input.size() + deps.size());
        for (auto& i : input) result.push_back(i);
        for (auto& dep : deps) result.push_back({std::ref(const_cast<primitive_id&>(dep.first.get())), dep.second});
        return result;
    }

    /// @brief Returns copy of all primitive ids on which this primitive depends - inputs, weights, biases, etc.
    std::vector<input_info> dependencies() const {
        std::vector<input_info> result;
        auto deps = get_dependencies();
        result.reserve(input.size() + deps.size());
        for (auto& i : input) result.push_back(i);
        for (auto dep : deps) result.push_back({std::ref(dep.first.get()), dep.second});
        return result;
    }

    virtual primitive_id type_string() const = 0;

    /// @brief Implicit conversion to primiitive id.
    operator primitive_id() const { return id; }

    /// @brief Primitive's type id.
    const primitive_type_id type;

    /// @brief Primitive's id.
    const primitive_id id;

    /// @brief Primitive's external id.
    const primitive_id ext_prim_id;

    /// @brief Requested output padding.
    std::vector<padding> output_paddings;

    /// @brief Requested output precision, if any.
    std::vector<optional_data_type> output_data_types;

    size_t input_size() const { return input.size(); }

    size_t output_size() const { return num_outputs; }

    using primitive_id_arr = std::vector<primitive_id>;

    /// @brief List of ids of input primitives.
    std::vector<input_info> input;

    int num_outputs;

protected:
#if 0 // TODO(taylor) to remove
    virtual std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const { return {}; }
#endif
    virtual std::vector<std::pair<std::reference_wrapper<const primitive_id>, int>> get_dependencies() const {
        // TODO(taylor) fix at inhereted primitives
        return {};
    }

    class condition;
    friend struct primitive_info;
};

/// @brief base class for all primitives implementations.
template <class PType>
class primitive_base : public primitive {
protected:
    explicit primitive_base(const primitive_id& id,
                            const std::vector<input_info>& input,
                            const primitive_id& ext_prim_id = "",
                            const std::vector<padding>& output_paddings = {padding()},
                            std::vector<optional_data_type> output_data_types = {optional_data_type()},
                            const int num_outputs = 1)
        : primitive(PType::type_id(), id, input, ext_prim_id, output_paddings, output_data_types, num_outputs) {}
};

// TODO(taylor) not completed yet
struct primitive_info {
    primitive_info(const primitive_id& original_id,
                   const std::string& type_id,
                   const std::vector<input_info>& dependencies,
                   const std::vector<primitive_id>& users, // TODO (taylor) multiple output
                   const std::vector<primitive_id>& fused_ids,
                   const std::vector<layout>& output_layouts,
                   const std::vector<std::string>& layout_strs,
                   const std::string& kernel_id,
                   const data_types& runtime_precision,
                   bool is_cpu,
                   int exec_id)
        : original_id(original_id),
          type_id(type_id),
          c_dependencies(dependencies),
          c_users(users),
          c_fused_ids(fused_ids),
          output_layouts(output_layouts),
          layout_strs(layout_strs),
          kernel_id(kernel_id),
          runtime_precision(runtime_precision),
          is_cpu(is_cpu),
          exec_id(exec_id) {}

    primitive_id original_id;
    std::string type_id;
    std::vector<input_info> c_dependencies;
    primitive::primitive_id_arr c_users;
    primitive::primitive_id_arr c_fused_ids;
    std::vector<layout> output_layouts;
    std::vector<std::string> layout_strs;
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
