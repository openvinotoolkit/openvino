// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ternocl_int2_weights_reorder.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>

#include "fully_connected_ternocl_int2.hpp"
#include "impls/cpu/cpu_impl_helpers.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "openvino/core/parallel.hpp"
#include "primitive_inst.h"
#include "registry/registry.hpp"
#include "reorder_inst.h"

namespace cldnn {
namespace ocl {

namespace {

// [N, K] u2 codes -> [K/16, N] uint32, re-encoding (code - 1) to the kernel's
// two's-complement {0, +1, -1} = {0, 1, 3}; row 16*kb+j of column n sits at bits [2j, 2j+1].
// The TernOCL FC impl only requests this layout for a scalar zero point of 1.
// u2 is little-endian inside each byte, so on a little-endian host the 16 codes of one word are
// 4 consecutive source bytes already at their output bit positions: each 2-bit field b1b0 maps
// 00->11, 01->00, 10->01.
// Returns false if a code is 3, which is not ternary.
bool pack_weights(const uint8_t* src, uint32_t* dst, size_t N, size_t K) {
    constexpr uint32_t lo = 0x55555555u;
    const size_t kblocks = K / kTernoclPackFactor;
    std::atomic<bool> ternary{true};
    ov::parallel_for(kblocks, [&](size_t kb) {
        uint32_t invalid = 0;
        for (size_t n = 0; n < N; ++n) {
            uint32_t w;
            std::memcpy(&w, src + (n * K + kb * kTernoclPackFactor) / 4, sizeof(w));
            const uint32_t b0 = w & lo;
            const uint32_t b1 = (w >> 1) & lo;
            invalid |= b0 & b1;
            dst[kb * N + n] = (~b0 & lo) | ((~(b0 ^ b1) & lo) << 1);
        }
        if (invalid != 0)
            ternary = false;
    });
    return ternary;
}

}  // namespace

struct ternocl_int2_weights_reorder : public typed_primitive_impl<reorder> {
    using parent = typed_primitive_impl<reorder>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::ternocl_int2_weights_reorder)

    // Used by deserialization, whose load() restores the kernel name.
    ternocl_int2_weights_reorder() = default;
    explicit ternocl_int2_weights_reorder(const std::string& kernel_name) : parent(kernel_name) {}

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<ternocl_int2_weights_reorder>(*this);
    }

    bool is_cpu() const override {
        return true;
    }

    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}
    void update(primitive_inst&, const kernel_impl_params&) override {}

    event::ptr execute_impl(const std::vector<event::ptr>& events, reorder_inst& instance) override {
        auto& stream = instance.get_network().get_stream();
        stream.wait_for_events(events);

        const auto& in_layout = instance.get_impl_params()->get_input_layout(0);
        const auto& shape = in_layout.get_shape();
        const size_t N = shape[0];
        const size_t K = shape[1];
        OPENVINO_ASSERT(ov::shape_size(shape) == N * K && K % kTernoclPackFactor == 0,
                        "[GPU] ternocl int2: unexpected weights shape ",
                        in_layout.to_short_string(),
                        " for ",
                        instance.id());

        cldnn::mem_lock<uint8_t, mem_lock_type::read> src(instance.input_memory_ptr(), stream);
        cldnn::mem_lock<uint32_t, mem_lock_type::write> dst(instance.output_memory_ptr(), stream);
        OPENVINO_ASSERT(src.size() >= N * K / 4 && dst.size() * sizeof(uint32_t) >= N * K / 4,
                        "[GPU] ternocl int2: weight buffers are smaller than ",
                        N * K / 4,
                        " B for ",
                        instance.id());
        OPENVINO_ASSERT(pack_weights(src.data(), dst.data(), N, K),
                        "[GPU] ternocl int2: ",
                        instance.id(),
                        " has a weight code outside the ternary range {-1, 0, +1}");

        return cpu::make_output_event(stream, instance.is_output());
    }
};

// A forced impl type (unit tests) skips validate() and takes the first manager of that type,
// so reorders other than the TernOCL weights packing are handed to the next manager of the type.
static const ImplementationManager& stock_reorder_manager(const ImplementationManager& self, shape_types shape_type) {
    for (const auto& m : ov::intel_gpu::Registry<reorder>::get_implementations()) {
        if (m->get_impl_type() == self.get_impl_type() && m->get_type_info() != self.get_type_info() && (m->get_shape_type() & shape_type) == shape_type)
            return *m;
    }
    OPENVINO_THROW("[GPU] ternocl int2: no reorder implementation to fall back to");
}

std::unique_ptr<primitive_impl> TernoclInt2WeightsReorderImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    if (!validate_impl(node))
        return stock_reorder_manager(*this, get_shape_type(node)).create_impl(node, params);
    return std::make_unique<ternocl_int2_weights_reorder>("ternocl_int2_weights_reorder");
}

std::unique_ptr<primitive_impl> TernoclInt2WeightsReorderImplementationManager::create_impl(const kernel_impl_params& params) const {
    if (params.get_output_layout().format != format::is_os_yx_isv16_ternary)
        return stock_reorder_manager(*this, get_shape_type(params)).create_impl(params);
    return std::make_unique<ternocl_int2_weights_reorder>("ternocl_int2_weights_reorder");
}

in_out_fmts_t TernoclInt2WeightsReorderImplementationManager::query_formats(const program_node& node) const {
    if (!validate_impl(node))
        return stock_reorder_manager(*this, get_shape_type(node)).query_formats(node);
    return ImplementationManager::query_formats(node);
}

}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::ternocl_int2_weights_reorder)
