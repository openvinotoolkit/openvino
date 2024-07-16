// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include <vector>
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/graph/serialization/vector_serializer.hpp"

namespace cldnn {

/// @brief reorder mean operation modes
enum class reorder_mean_mode {
    none,      // val
    subtract,  // val - mean
    mul,       // val * mean
    div,       // val/mean
};

struct WeightsReorderParams {
    WeightsReorderParams() {}

    WeightsReorderParams(const layout& in_layout, const layout& out_layout, bool transposed = false, bool grouped = false)
        : _in_layout(in_layout),
          _out_layout(out_layout),
          _transposed(transposed),
          _grouped(grouped) {}

    size_t hash() const {
        size_t seed = hash_combine(_in_layout.hash(), _out_layout.hash());
        seed = hash_combine(seed, _transposed);
        seed = hash_combine(seed, _grouped);
        return seed;
    }

    bool operator==(const WeightsReorderParams& rhs) const {
        if (typeid(*this) != typeid(rhs))
            return false;

        return _in_layout == rhs._in_layout &&
               _out_layout == rhs._out_layout &&
               _transposed == rhs._transposed &&
               _grouped == rhs._grouped;
    }

    layout get_input_layout() const { return _in_layout; }
    layout get_output_layout() const { return _out_layout; }
    bool should_be_transposed() const { return _transposed; }
    bool get_grouped() const { return _grouped; }

    void set_input_layout(const layout& layout) { _in_layout = layout; }
    void set_output_layout(const layout& layout) { _out_layout = layout; }

    void save(cldnn::BinaryOutputBuffer& ob) const {
        ob << _in_layout;
        ob << _out_layout;
        ob << _transposed;
        ob << _grouped;
    }
    void load(cldnn::BinaryInputBuffer& ib) {
        ib >> _in_layout;
        ib >> _out_layout;
        ib >> _transposed;
        ib >> _grouped;
    }
    virtual ~WeightsReorderParams() = default;

protected:
    layout _in_layout;
    layout _out_layout;
    bool _transposed;
    bool _grouped;
};

/// @brief Changes how data is ordered in memory. Value type is not changed & all information is preserved.
/// @details Corresponding values are bitwise equal before/after reorder.
/// Also merged with subtraction layer, which can subtract, multiply or divide values based on mean_mode value, while doing reordering.
/// NOTE THAT THIS WILL SUBTRACT THE SAME VALUES FROM EACH BATCH.
struct reorder : public primitive_base<reorder> {
    CLDNN_DECLARE_PRIMITIVE(reorder)

    reorder() : primitive_base("", {}),
                output_format(format::any),
                mean_mode(reorder_mean_mode::subtract) {}

    /// @brief reorder memory types
    enum class memory_type {
        buffer,
        surface
    };

    /// @brief Constructs reorder primitive with directly provided mean subtract values.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param output_layout Requested memory layout.
    /// @param values_to_subtract Array of mean subtract values.
    reorder(const primitive_id& id,
            const input_info& input,
            const layout& output_layout,
            const std::vector<float>& values_to_subtract = {},
            const reorder_mean_mode mode = reorder_mean_mode::subtract)
        : primitive_base(id, {input}, 1, {optional_data_type {output_layout.data_type}}, {output_layout.data_padding}),
          output_format(output_layout.format),
          mean(""),
          subtract_per_feature(values_to_subtract),
          mean_mode(mode) {}

    /// @brief Constructs reorder primitive which takes mean subtract values from another primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param output_layout Requested memory layout.
    /// @param mean Primitive id to get mean subtract values.
    reorder(const primitive_id& id,
            const input_info& input,
            const layout& output_layout,
            primitive_id const& mean,
            const reorder_mean_mode mode = reorder_mean_mode::subtract)
        : primitive_base(id, {input}, 1, {optional_data_type {output_layout.data_type}}, {output_layout.data_padding}),
          output_format(output_layout.format),
          mean(mean),
          subtract_per_feature(0),
          mean_mode(mode) {}

    /// @brief Constructs reorder primitive with directly provided mean subtract values.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param output_layout Requested memory layout.
    /// @param values_to_subtract Array of mean subtract values.
    /// @param truncate Convert truncation mode.
    reorder(const primitive_id& id,
            const input_info& input,
            format output_format,
            data_types output_data_type,
            const std::vector<float>& values_to_subtract = {},
            const reorder_mean_mode mode = reorder_mean_mode::subtract,
            const padding& output_padding = padding(),
            const bool truncate = false)
        : primitive_base(id, {input}, 1, {optional_data_type{output_data_type}},  {output_padding}),
          output_format(output_format),
          mean(""),
          subtract_per_feature(values_to_subtract),
          mean_mode(mode),
          truncate(truncate) {}

    /// @brief Constructs reorder primitive which takes mean subtract values from another primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param output_layout Requested memory layout.
    /// @param mean Primitive id to get mean subtract values.
    reorder(const primitive_id& id,
            const input_info& input,
            format output_format,
            data_types output_data_type,
            primitive_id const& mean,
            const reorder_mean_mode mode = reorder_mean_mode::subtract,
            const padding& output_padding = padding())
        : primitive_base(id, {input}, 1, {optional_data_type {output_data_type}}, {output_padding}),
          output_format(output_format),
          mean(mean),
          subtract_per_feature(0),
          mean_mode(mode) {}

    /// @brief Constructs reorder primitive with two inputs and directly provided mean subtract values.
    /// @param id This primitive id.
    /// @param input input primitive id.
    /// @param input input2 primitive id.
    /// @param output_layout Requested memory layout.
    /// @param values_to_subtract Array of mean subtract values.
    reorder(const primitive_id& id,
            const input_info& input,
            const input_info& input2,
            const layout& output_layout,
            const std::vector<float>& values_to_subtract = {},
            const reorder_mean_mode mode = reorder_mean_mode::subtract)
        : primitive_base(id, { input, input2 }, 1, {optional_data_type { output_layout.data_type }}, {output_layout.data_padding}),
          output_format(output_layout.format),
          mean(""),
          subtract_per_feature(values_to_subtract),
          mean_mode(mode) {}

    /// @brief Constructs reorder primitive with two inputs, which takes mean subtract values from another primitive.
    /// @param id This primitive id.
    /// @param input input primitive id.
    /// @param input input2 primitive id.
    /// @param output_layout Requested memory layout.
    /// @param mean Primitive id to get mean subtract values.
    reorder(const primitive_id& id,
            const input_info& input,
            const input_info& input2,
            const layout& output_layout,
            primitive_id const& mean,
            const reorder_mean_mode mode = reorder_mean_mode::subtract)
        : primitive_base(id, { input, input2 }, 1, {optional_data_type{ output_layout.data_type }}, {output_layout.data_padding}),
        output_format(output_layout.format),
        mean(mean),
        mean_mode(mode) {}

    /// @brief Constructs weights reorder primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights_reorder_params Parameters required for reorder weights.
    reorder(const primitive_id& id,
            const input_info& input,
            std::shared_ptr<WeightsReorderParams> weights_reorder_params)
        : primitive_base(id, {input}),
          output_format(weights_reorder_params->get_output_layout().format),
          mean(""),
          subtract_per_feature({}),
          mean_mode(reorder_mean_mode::none),
          weights_reorder_params(weights_reorder_params) {}

    /// @brief Requested memory format.
    format output_format;
    /// @brief Primitive id to get mean subtract values. Ignored if subtract_per_feature is set.
    primitive_id mean;
    /// @brief Array of mean subtract values.
    std::vector<float> subtract_per_feature;
    /// @brief Mode of mean execution.
    reorder_mean_mode mean_mode;
    /// @brief Input memory type.
    memory_type input_mem_type = memory_type::buffer;
    /// @brief Parameters required for reorder weights.
    std::shared_ptr<WeightsReorderParams> weights_reorder_params = {};

    inline bool has_surface_input() const {
        return input.size() == 1 &&
               input_mem_type == memory_type::surface;
    }

    /// @brief Convert truncation Mode
    bool truncate = false;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, mean_mode);
        seed = hash_combine(seed, input_mem_type);
        seed = hash_combine(seed, truncate);
        seed = hash_range(seed, subtract_per_feature.begin(), subtract_per_feature.end());
        seed = hash_combine(seed, mean.empty());

        if (weights_reorder_params) {
            seed = hash_combine(seed, weights_reorder_params->hash());
        }
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const reorder>(rhs);

        bool reorder_weights_eq = (weights_reorder_params == nullptr) == (rhs_casted.weights_reorder_params == nullptr);
        if (reorder_weights_eq && weights_reorder_params) {
            reorder_weights_eq = *weights_reorder_params == *rhs_casted.weights_reorder_params;
        }

        return subtract_per_feature == rhs_casted.subtract_per_feature &&
               mean_mode == rhs_casted.mean_mode &&
               input_mem_type == rhs_casted.input_mem_type &&
               truncate == rhs_casted.truncate &&
               output_format == rhs_casted.output_format &&
               mean.empty() == rhs_casted.mean.empty() &&
               reorder_weights_eq;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<reorder>::save(ob);
        ob << output_format;
        ob << mean;
        ob << subtract_per_feature;
        ob << make_data(&mean_mode, sizeof(reorder_mean_mode));
        ob << make_data(&input_mem_type, sizeof(memory_type));
        if (weights_reorder_params == nullptr) {
            ob << false;
        } else {
            ob << true;
            weights_reorder_params->save(ob);
        }
        ob << truncate;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<reorder>::load(ib);
        ib >> output_format;
        ib >> mean;
        ib >> subtract_per_feature;
        ib >> make_data(&mean_mode, sizeof(reorder_mean_mode));
        ib >> make_data(&input_mem_type, sizeof(memory_type));
        bool has_weights_reorder_params;
        ib >> has_weights_reorder_params;
        if (has_weights_reorder_params) {
            weights_reorder_params = std::make_shared<WeightsReorderParams>();
            weights_reorder_params->load(ib);
        }
        ib >> truncate;
    }

protected:
    std::vector<input_info> get_dependencies() const override {
        if (mean.empty())
            return {};
        return {mean};
    }
};

}  // namespace cldnn
