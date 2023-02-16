// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/graph/serialization/binary_buffer.hpp"
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

/// @brief Globally unique primitive's type id
using primitive_type_id = struct primitive_type *;

/// @brief Unique @p id of a primitive within a topology.
using primitive_id = std::string;

struct primitive_info;

/// @brief Describes information of inputs.
/// @details Contains infomation about id and output index of input primitive.
struct input_info {
    input_info() : pid(""), idx(0) {}
    input_info(primitive_id pid) : pid(pid), idx(0) {}
    input_info(primitive_id pid, int idx) : pid(pid), idx(idx) {}

    /// @brief Copy assignment.
    input_info& operator=(const input_info& other) {
        if (this == &other)
            return *this;
        pid = other.pid;
        idx = other.idx;
        return *this;
    }

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
              const std::vector<padding>& output_paddings = {padding()},
              const std::vector<optional_data_type> output_data_types = {optional_data_type()},
              const size_t num_outputs = 1)
        : type(type),
          id(id),
          output_paddings(output_paddings),
          output_data_types(output_data_types),
          input(input),
          num_outputs(num_outputs) {}

    virtual ~primitive() = default;

    /// @brief Returns copy of all input info on which this primitive depends - inputs, weights, biases, etc.
    std::vector<input_info> dependencies() const {
        auto result = input;
        auto deps = get_dependencies();
        for (auto& pid : deps) result.push_back({pid, 0});
        return result;
    }

    virtual primitive_id type_string() const = 0;

    virtual size_t hash() const {
        size_t seed = 0;
        // hash for type
        primitive_id type_str = type_string();
        for (size_t idx = 0; idx < type_str.size(); idx++) {
            seed = hash_combine(seed, type_str[idx]);
        }

        // hash for number of outputs
        seed = hash_combine(seed, num_outputs);

        // hash for number of inputs
        auto inputs = dependencies();
        seed = hash_combine(seed, inputs.size());
        return seed;
    }

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
    std::vector<padding> output_paddings;

    /// @brief Requested output precision, if any.
    std::vector<optional_data_type> output_data_types;

    size_t input_size() const { return input.size(); }

    size_t output_size() const { return num_outputs; }

    using primitive_id_arr = std::vector<primitive_id>;

    using input_info_arr = std::vector<input_info>;

    /// @brief List of input info containing id and output index of input primitive.
    input_info_arr input;

    size_t num_outputs;

    virtual std::string get_type() const { return "NONE"; }
    virtual void save(BinaryOutputBuffer& ob) const { }
    virtual void load(BinaryInputBuffer& ib) { }

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
                            const std::vector<input_info>& input,
                            const std::vector<padding>& output_paddings = {padding()},
                            const std::vector<optional_data_type> output_data_types = {optional_data_type()},
                            const size_t num_outputs = 1)
        : primitive(PType::type_id(), id, input, output_paddings, output_data_types, num_outputs) {}
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

#define GPU_DEFINE_PRIMITIVE_TYPE_ID(PType)             \
    primitive_type_id PType::type_id() {                \
        static primitive_type_base<PType> instance;     \
        return &instance;                               \
    }                                                   \
    bool _##PType##_added_ = prim_map_storage::instance().set_type_id(#PType, PType::type_id());

struct prim_map_storage {
    static prim_map_storage& instance() {
        static prim_map_storage instance;
        return instance;
    }

    const cldnn::primitive_type_id get_type_id(const std::string& type_string) const {
        return map.at(type_string);
    }

    bool set_type_id(const std::string& type_string, const cldnn::primitive_type_id type_id) {
        return map.insert({type_string, type_id}).second;
    }

private:
    std::unordered_map<std::string, cldnn::primitive_type_id> map;
};
}  // namespace cldnn
