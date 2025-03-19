// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/graph/serialization/layout_serializer.hpp"
#include "intel_gpu/graph/serialization/set_serializer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/graph/serialization/tensor_serializer.hpp"
#include "intel_gpu/graph/serialization/vector_serializer.hpp"
#include "intel_gpu/runtime/compounds.hpp"
#include "intel_gpu/runtime/layout.hpp"

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
    input_info(primitive_id pid) : pid(std::move(pid)), idx(0) {}
    input_info(primitive_id pid, int idx) : pid(std::move(pid)), idx(idx) {}

    /// @brief Copy assignment.
    input_info& operator=(const input_info& other) {
        if (this == &other)
            return *this;
        pid = other.pid;
        idx = other.idx;
        return *this;
    }

    /// @brief Compare
    bool operator==(const input_info& rhs) const {
        return ((pid == rhs.pid) && (idx == rhs.idx));
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

    bool is_valid() const {
        return pid.compare("") != 0;
    }

    void save(BinaryOutputBuffer& ob) const {
        ob << pid;
        ob << idx;
    }

    void load(BinaryInputBuffer& ib) {
        ib >> pid;
        ib >> idx;
    }

    std::string to_string() const {
        std::stringstream ss;
        ss << "input_info(pid:" << pid << ",idx:" << idx << ")";
        return ss.str();
    }
};

static inline std::ostream& operator<< (std::ostream& os, input_info& info) {
    os << info.to_string();
    return os;
}

struct prim_map_storage {
    static prim_map_storage& instance() {
        static prim_map_storage instance;
        return instance;
    }

    const cldnn::primitive_type_id get_type_id(const std::string& type_string) const {
        return map.at(type_string);
    }

    const cldnn::primitive_id get_type_string(const cldnn::primitive_type_id type_id) const {
        return inverse_map.at(type_id);
    }

    bool set_type_id(const std::string& type_string, const cldnn::primitive_type_id type_id) {
        return map.insert({type_string, type_id}).second && inverse_map.insert({type_id, type_string}).second;
    }

private:
    std::unordered_map<std::string, cldnn::primitive_type_id> map;
    std::unordered_map<cldnn::primitive_type_id, std::string> inverse_map;
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
          num_outputs(num_outputs) {
        if (output_paddings.size() < num_outputs) {
            this->output_paddings.insert(this->output_paddings.end(), num_outputs - output_paddings.size(), padding());
        }
        if (output_data_types.size() < num_outputs) {
            this->output_data_types.insert(this->output_data_types.end(), num_outputs - output_data_types.size(), optional_data_type());
        }
    }

    virtual ~primitive() = default;

    /// @brief Returns copy of all input info on which this primitive depends - inputs, weights, biases, etc.
    std::vector<input_info> dependencies() const {
        auto result = input;
        auto deps = get_dependencies();
        for (auto& dep : deps) result.push_back(dep);
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

    bool compare_common_params(const primitive& rhs) const {
        if (type != rhs.type)
            return false;

        if (num_outputs != rhs.num_outputs)
            return false;

        if (dependencies().size() != rhs.dependencies().size())
            return false;

        if (output_data_types.size() != rhs.output_data_types.size())
            return false;

        for (size_t i = 0; i < output_data_types.size(); ++i) {
            if (output_data_types[i].value_or(data_types::dynamic) !=
                rhs.output_data_types[i].value_or(data_types::dynamic))
                return false;
        }

        if (output_paddings.size() != rhs.output_paddings.size())
            return false;

        for (size_t i = 0; i < output_paddings.size(); ++i) {
            if (output_paddings[i] != rhs.output_paddings[i])
                return false;
        }

        return true;
    }

    virtual bool operator==(const primitive& rhs) const { return false; }

    bool operator!=(const primitive& rhs) const { return !(*this == rhs); }

    /// @brief Implicit conversion to primitive id.
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

    virtual const std::string& get_type_info() const = 0;
    virtual void save(BinaryOutputBuffer& ob) const {
        ob << type_string();
        ob << id;
        ob << origin_op_name;
        ob << origin_op_type_name;
        ob << output_paddings;
        ob << output_data_types.size();
        for (auto& output_data_type : output_data_types) {
            if (output_data_type.has_value()) {
                ob << true;
                ob << make_data(&output_data_type.value(), sizeof(data_types));
            } else {
                ob << false;
            }
        }
        ob << input;
        ob << num_outputs;
    }

    virtual void load(BinaryInputBuffer& ib) {
        std::string type_str;
        ib >> type_str;
        *const_cast<primitive_type_id*>(&type) = prim_map_storage::instance().get_type_id(type_str);
        ib >> *const_cast<primitive_id*>(&id);
        ib >> origin_op_name;
        ib >> origin_op_type_name;
        ib >> output_paddings;
        size_t output_data_types_size;
        ib >> output_data_types_size;
        output_data_types.clear();
        for (size_t i = 0; i < output_data_types_size; i++) {
            bool has_value;
            ib >> has_value;
            if (has_value) {
                data_types data_type = data_types();
                ib >> make_data(&data_type, sizeof(data_types));
                output_data_types.emplace_back(optional_data_type(data_type));
            } else {
                output_data_types.emplace_back(optional_data_type());
            }
        }
        ib >> input;
        ib >> num_outputs;
    }

    virtual padding get_output_padding(size_t idx) const {
        if (idx < output_paddings.size()) {
            return output_paddings[idx];
        } else {
            return padding();
        }
    }

    virtual optional_data_type get_output_data_type(size_t idx) const {
        if (idx < output_data_types.size()) {
            return output_data_types[idx];
        } else {
            return optional_data_type();
        }
    }

protected:
    virtual std::vector<input_info> get_dependencies() const { return {}; }
    class condition;
    friend struct primitive_info;
};

/// @brief base class for all primitives implementations.
template <class PType>
class primitive_base : public primitive {
protected:
    explicit primitive_base(const primitive_id& id,
                            const std::vector<input_info>& input,
                            const size_t num_outputs = 1,
                            const std::vector<optional_data_type> output_data_types = {optional_data_type()},
                            const std::vector<padding>& output_paddings = {padding()})
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
    DECLARE_OBJECT_TYPE_SERIALIZATION(PType) \
    CLDNN_DEFINE_TYPE_ID(PType)              \
    CLDNN_DEFINE_TYPE_STRING(PType)

#define GPU_DEFINE_PRIMITIVE_TYPE_ID(PType)             \
    primitive_type_id PType::type_id() {                \
        static primitive_type_base<PType> instance;     \
        return &instance;                               \
    }                                                   \
    bool _##PType##_added_ = prim_map_storage::instance().set_type_id(#PType, PType::type_id());
}  // namespace cldnn
