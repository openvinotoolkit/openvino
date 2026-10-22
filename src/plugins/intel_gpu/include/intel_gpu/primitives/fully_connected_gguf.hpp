// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/op/gguf_dequantize.hpp"
#include "primitive.hpp"

namespace cldnn {

/// @brief Fully connected layer on GGUF k-quant weights.
/// @details
///   input   : [..., M, K], f16 or f32
///   weights : u8 [N, K / 256 * block_bytes], the GGUF blocks of N rows (see ov::intel_gpu::op::FullyConnectedGGUF)
///   output  : [..., M, N]
/// It is not a fully_connected because its weights are not [N, K].
struct fully_connected_gguf : public primitive_base<fully_connected_gguf> {
    CLDNN_DECLARE_PRIMITIVE(fully_connected_gguf)

    using Format = ov::op::internal::GGUFDequantize::Format;

    fully_connected_gguf() : primitive_base("", {}) {}

    fully_connected_gguf(const primitive_id& id, const input_info& input, const input_info& weights, Format format, const data_types output_data_type)
        : primitive_base(id, {input, weights}, 1, {optional_data_type{output_data_type}}),
          format(format) {}

    Format format = Format::Q6_K;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, static_cast<int>(format));
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs)) {
            return false;
        }
        return format == downcast<const fully_connected_gguf>(rhs).format;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<fully_connected_gguf>::save(ob);
        ob << make_data(&format, sizeof(format));
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<fully_connected_gguf>::load(ib);
        ib >> make_data(&format, sizeof(format));
    }
};
}  // namespace cldnn
