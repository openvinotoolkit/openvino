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

/// @brief Base class of network primitive description.
struct primitive {
public:
    /// @brief Initialize fields common for all primitives.
    primitive(const primitive_type_id& type,
              const primitive_id& id,
              const std::vector<primitive_id>& input,
              const primitive_id& ext_prim_id = "",
              const padding& output_padding = padding(),
              const optional_data_type output_data_type = optional_data_type())
        : type(type),
          id(id),
          ext_prim_id(ext_prim_id),
          output_padding(output_padding),
          output_data_type(output_data_type),
          input(input) {}

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
    padding output_padding;

    /// @brief Requested output precision, if any.
    optional_data_type output_data_type;

    size_t input_size() const { return input.size(); }

    using primitive_id_arr = std::vector<primitive_id>;

    /// @brief List of ids of input primitives.
    primitive_id_arr input;

protected:
    virtual std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const { return {}; }
    class condition;
    friend struct primitive_info;
};

/// @brief base class for all primitives implementations.
template <class PType>
class primitive_base : public primitive {
protected:
    explicit primitive_base(const primitive_id& id,
                            const std::vector<primitive_id>& input,
                            const primitive_id& ext_prim_id = "",
                            const padding& output_padding = padding(),
                            optional_data_type output_data_type = optional_data_type())
        : primitive(PType::type_id(), id, input, ext_prim_id, output_padding, output_data_type) {}
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
