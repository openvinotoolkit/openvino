// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "turboq_quantize.hpp"

namespace ov::Extensions::Cpu::XARCH {

void turboq_quantize_head(const void* src,
                          void* dst,
                          float* out_norm,
                          int head_dim,
                          int bits,
                          ov::element::Type src_precision) {
    if (g_turboq_fused_quantize && src_precision == ov::element::f32) {
        turboq_quantize_head_fused(static_cast<const float*>(src), dst, out_norm, head_dim, bits);
        return;
    }
    assert((bits == 3 || bits == 4) && "bits must be 3 or 4");
    assert(head_dim % 64 == 0 && "head_dim must be divisible by 64");

    const int dim = head_dim;
    const float sqrt_dim = std::sqrt(static_cast<float>(dim));

    std::vector<float> unit(dim);
    const float norm = dispatch_normalize(src, unit.data(), dim, src_precision);

    std::vector<float> rotated(dim);
    turboq_rotate_forward(unit.data(), rotated.data(), dim);
    for (int i = 0; i < dim; i++) {
        rotated[i] *= sqrt_dim;
    }

    std::vector<uint8_t> indices(dim);
    const float* boundaries = turboq_boundaries(bits, dim);
    int n_boundaries = (bits == 4) ? 15 : 7;
    for (int i = 0; i < dim; i++) {
        indices[i] = scalar_quantize(rotated[i], boundaries, n_boundaries);
    }

    auto* out = static_cast<uint8_t*>(dst);
    if (bits == 4) {
        pack_4bit(indices.data(), out, dim);
    } else {
        pack_3bit(indices.data(), out, dim);
    }

    float stored_norm = norm;
    if (g_turboq_norm_correction) {
        const float* codebook = turboq_codebook(bits, dim);
        float recon_sq = 0.0F;
        for (int i = 0; i < dim; i++) {
            float c = codebook[indices[i]];
            recon_sq += c * c;
        }
        float recon_norm = std::sqrt(recon_sq);
        if (recon_norm > 1e-30F) {
            stored_norm = norm * sqrt_dim / recon_norm;
        }
    }
    *out_norm = stored_norm;
}

}  // namespace ov::Extensions::Cpu::XARCH
