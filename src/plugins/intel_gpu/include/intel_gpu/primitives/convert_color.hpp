// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"

namespace cldnn {

/// @brief Performs image conversion from one format to another
struct convert_color : public primitive_base<convert_color> {
    CLDNN_DECLARE_PRIMITIVE(convert_color)

    convert_color() : primitive_base("", {}) {}

    enum color_format : uint32_t {
        RGB,       ///< RGB color format
        BGR,       ///< BGR color format, default in OpenVINO
        RGBX,      ///< RGBX color format with X ignored during inference
        BGRX,      ///< BGRX color format with X ignored during inference
        NV12,      ///< NV12 color format represented as compound Y+UV blob
        I420,      ///< I420 color format represented as compound Y+U+V blob
    };

    enum memory_type : uint32_t {
        buffer,
        image
    };

    /// @brief Constructs convert_color primitive.
    /// @param id This primitive id.
    /// @param inputs Input primitives ids.
    /// @param input_color_format Color to convert from.
    /// @param output_color_format Color to convert to.
    /// @param mem_type Memory type.
    convert_color(const primitive_id& id,
                  const std::vector<input_info>& inputs,
                  const color_format input_color_format,
                  const color_format output_color_format,
                  const memory_type mem_type)
        : primitive_base(id, inputs),
          input_color_format(input_color_format),
          output_color_format(output_color_format),
          mem_type(mem_type) {}

    color_format input_color_format = color_format::RGB;
    color_format output_color_format = color_format::RGB;
    memory_type mem_type = memory_type::buffer;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, input_color_format);
        seed = hash_combine(seed, output_color_format);
        seed = hash_combine(seed, mem_type);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const convert_color>(rhs);

        return input_color_format == rhs_casted.input_color_format &&
               output_color_format == rhs_casted.output_color_format &&
               mem_type == rhs_casted.mem_type;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<convert_color>::save(ob);
        ob << make_data(&input_color_format, sizeof(color_format));
        ob << make_data(&output_color_format, sizeof(color_format));
        ob << make_data(&mem_type, sizeof(memory_type));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<convert_color>::load(ib);
        ib >> make_data(&input_color_format, sizeof(color_format));
        ib >> make_data(&output_color_format, sizeof(color_format));
        ib >> make_data(&mem_type, sizeof(memory_type));
    }
};
}  // namespace cldnn
