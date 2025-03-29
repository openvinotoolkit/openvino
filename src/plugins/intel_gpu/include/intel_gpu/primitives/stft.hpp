// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Short time fourier transform (STFT) operation.
/// @details Checks the specification for details.
struct STFT : public primitive_base<STFT> {
    CLDNN_DECLARE_PRIMITIVE(STFT)

    STFT() : primitive_base("", {}) {}

    /// @brief Constructs STFT primitive.
    /// @param id This primitive id.
    /// @param signal signal input.
    /// @param window window input.
    /// @param frame_size Size of the frame.
    /// @param frame_step Step between frames.
    /// @param transpose_frames Enable/Disable transpose_frames(check specification for details)..

    STFT(const primitive_id& id,
         const input_info& signal,
         const input_info& window,
         const input_info& frame_size,
         const input_info& frame_step,
         const bool transpose_frames)
        : primitive_base(id, {signal, window, frame_size, frame_step}),
          transpose_frames(transpose_frames) {}

    /// @brief Enable/Disabletranspose_frames(check specification for details).
    bool transpose_frames = false;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, transpose_frames);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const STFT>(rhs);

        return transpose_frames == rhs_casted.transpose_frames;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<STFT>::save(ob);
        ob << transpose_frames;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<STFT>::load(ib);
        ib >> transpose_frames;
    }
};
}  // namespace cldnn
