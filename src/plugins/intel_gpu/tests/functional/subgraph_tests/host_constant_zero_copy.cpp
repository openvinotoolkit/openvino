// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// On an integrated GPU, a large constant that owns page-aligned host data of the type the plugin uses is wrapped as a
// host buffer instead of being copied. The same weights at an unaligned address are copied as before. Both models
// must give the same results, and the wrapped one must allocate (almost) nothing for the weights on the device.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"

namespace {

constexpr size_t K = 1024, N = 1024;  // f16 weights: 2 MiB
constexpr size_t kPage = 1 << 16;     // a multiple of any system page size

// usm_host bytes allocated by the plugin (copied constants live there on an integrated GPU; wrapped ones do not)
uint64_t usm_host_bytes(ov::Core& core) {
    const auto stats = core.get_property("GPU", ov::intel_gpu::memory_statistics);
    const auto it = stats.find("usm_host");
    return it == stats.end() ? 0 : it->second;
}

// A model computing x + W, with W at `data` (which must stay alive while the compiled model is used). The constant
// is used as stored (no weight reorder), so its own allocation shows in the device memory statistics.
std::shared_ptr<ov::Model> make_model(void* data) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{static_cast<int64_t>(K), static_cast<int64_t>(N)});
    ov::Tensor w_t(ov::element::f16, ov::Shape{K, N}, data);
    auto w = std::make_shared<ov::op::v0::Constant>(w_t);
    return std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v1::Add>(x, w)}, ov::ParameterVector{x});
}

TEST(HostConstantZeroCopy, IntegratedGpuWrapsAlignedConstants) {
    ov::Core core;
    if (core.get_property("GPU", ov::device::type) != ov::device::Type::INTEGRATED) {
        GTEST_SKIP() << "integrated GPUs only";
    }
    const size_t bytes = K * N * sizeof(ov::float16);
    // the weights once at a page boundary (wrapped) and once 64 bytes past one (copied)
    auto aligned_storage = std::unique_ptr<char[]>(new char[bytes + kPage]);
    auto unaligned_storage = std::unique_ptr<char[]>(new char[bytes + 2 * kPage]);
    auto page_up = [](char* p) {
        return reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(p) + kPage - 1) / kPage * kPage);
    };
    char* aligned = page_up(aligned_storage.get());
    char* unaligned = page_up(unaligned_storage.get()) + 64;

    std::mt19937 rng(3);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    auto* w = reinterpret_cast<ov::float16*>(aligned);
    for (size_t i = 0; i < K * N; ++i) {
        w[i] = ov::float16(dist(rng));
    }
    std::memcpy(unaligned, aligned, bytes);

    ov::Tensor x(ov::element::f16, ov::Shape{K, N});
    for (size_t i = 0; i < x.get_size(); ++i) {
        x.data<ov::float16>()[i] = ov::float16(dist(rng) * 20.f);
    }

    const uint64_t base = usm_host_bytes(core);
    auto copied = core.compile_model(make_model(unaligned), "GPU");
    const uint64_t copied_bytes = usm_host_bytes(core) - base;
    auto wrapped = core.compile_model(make_model(aligned), "GPU");
    const uint64_t wrapped_bytes = usm_host_bytes(core) - base - copied_bytes;
    // the copied model allocates a copy of the weights; the wrapped one does not
    EXPECT_GE(copied_bytes, bytes);
    EXPECT_LT(wrapped_bytes + bytes / 2, copied_bytes) << "wrapped " << wrapped_bytes << " vs copied " << copied_bytes;

    auto run = [&](ov::CompiledModel& cm) {
        auto req = cm.create_infer_request();
        req.set_input_tensor(x);
        req.infer();
        ov::Tensor out(req.get_output_tensor().get_element_type(), req.get_output_tensor().get_shape());
        req.get_output_tensor().copy_to(out);
        return out;
    };
    const auto a = run(copied);
    const auto b = run(wrapped);
    ASSERT_EQ(a.get_byte_size(), b.get_byte_size());
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.get_byte_size()), 0) << "wrapped and copied weights give different results";
}

}  // namespace
