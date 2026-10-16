// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <openvino/core/type/float16.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/recurrent_cache.hpp>
#include <openvino/op/result.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#include <sstream>

namespace {
// Run changing B/T/maps/reset offsets through the same compiled graph and imported blob.
// Reference deliberately uses a reset copy, followed by gather and ordered row writes.
void run_cache_case(bool conv, ov::element::Type precision, ov::element::Type indices, bool fixed_shape = false,
                    bool transpose = false) {
    constexpr int S = 12, C = 7, H = 3, F = C * H;
    ov::ParameterVector params;
    ov::OutputVector args;
    auto add = [&](ov::element::Type type, ov::PartialShape shape) {
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        params.push_back(param);
        args.push_back(param);
    };
    add(precision, {1, 1, S, F});
    add(indices, {1, 1, 1, fixed_shape ? 1 : -1});
    add(indices, {1, 1, 1, fixed_shape ? 1 : -1});
    for (int i = 0; i < 2; ++i)
        add(indices, {1, 1, 1, 1});
    add(indices, {1, 1, 1, 2});
    if (conv) {
        add(precision, {1, fixed_shape ? 1 : -1, C, fixed_shape ? 1 : -1});
        add(indices, {1, 1, fixed_shape ? 3 : -1, 2});
    }
    std::shared_ptr<ov::Node> op;
    if (conv)
        op = std::make_shared<ov::op::internal::RecurrentConvCache>(args, H);
    else
        op = std::make_shared<ov::op::internal::RecurrentStateGather>(args, transpose ? H : 0, transpose ? C : 0);
    ov::ResultVector results;
    for (int i = 0; i < 2; ++i)
        results.push_back(std::make_shared<ov::op::v0::Result>(op->output(i)));
    ov::Core core;
    auto compiled = core.compile_model(std::make_shared<ov::Model>(results, params), "GPU");
    auto clone = op->clone_with_new_inputs(args);
    ASSERT_EQ(clone->get_output_partial_shape(0), op->get_output_partial_shape(0));
    std::stringstream blob;
    compiled.export_model(blob);
    auto imported = core.import_model(blob, "GPU");
    for (auto model : {compiled, imported}) {
        auto req = model.create_infer_request();
        for (int step = 0; step < (fixed_shape ? 1 : 8); ++step) {
            const int B = step % 4 + 1, T = step % 3 + 1;
            const int rb = step % S, rl = step % 2, eb = step % 3;
            std::vector<int> a(B), e = step % 2 ? std::vector<int>{4, 2, 4} : std::vector<int>{};
            for (int b = 0; b < B; ++b)
                a[b] = b == 0 ? rb : (step + b) % S;
            // Cycles: destination 0 reads row 1 and destination 1 reads row 0.
            if (step == 2) {
                e = {1, 0};
            }
            std::vector<std::pair<int, int>> windows;
            if (conv)
                for (int k = 0; k < 3; ++k)
                    windows.emplace_back(std::max(0, T - 2 + k), 8 - 4 * k);
            std::vector<float> cache(S * F), qkv(B * C * T);
            auto round = [&](float x) { return precision == ov::element::f16 ? float(ov::float16(x)) : x; };
            for (size_t i = 0; i < cache.size(); ++i)
                cache[i] = round(float(int(i % 89) - 40) / 8);
            for (size_t i = 0; i < qkv.size(); ++i)
                qkv[i] = round(float(int(i % 37) - 16) / 8);
            auto cleared = cache;
            for (int row = rb; row < rb + rl; ++row)
                std::fill_n(cleared.begin() + row * F, F, 0);
            auto expected_cache = cleared;
            for (size_t j = 0; j < e.size(); ++j)
                std::copy_n(cleared.begin() + e[j] * F, F, expected_cache.begin() + (eb + j) * F);
            std::vector<float> expected_active(conv ? B * C * (H + T) : B * F);
            for (int b = 0; b < B; ++b) {
                if (!conv)
                    std::copy_n(cleared.begin() + a[b] * F, F, expected_active.begin() + b * F);
                else
                    for (int c = 0; c < C; ++c) {
                        for (int t = 0; t < H + T; ++t)
                            expected_active[(b * C + c) * (H + T) + t] =
                                t < H ? cleared[a[b] * F + c * H + t] : qkv[(b * C + c) * T + t - H];
                    }
            }
            if (transpose) {
                auto plain = expected_active;
                for (int b = 0; b < B; ++b)
                    for (int k = 0; k < H; ++k)
                        for (int v = 0; v < C; ++v)
                            expected_active[b * F + k * C + v] = plain[b * F + v * H + k];
            }
            for (auto [start, dst] : windows)
                for (int b = 0; b < B; ++b)
                    for (int c = 0; c < C; ++c)
                        for (int t = 0; t < H; ++t)
                            expected_cache[(dst + b) * F + c * H + t] =
                                expected_active[(b * C + c) * (H + T) + start + t];
            auto data = [&](int index, ov::Shape shape, const std::vector<float> &values) {
                ov::Tensor tensor(precision, shape);
                for (size_t i = 0; i < values.size(); ++i) {
                    if (precision == ov::element::f16)
                        tensor.data<ov::float16>()[i] = ov::float16(values[i]);
                    else
                        tensor.data<float>()[i] = values[i];
                }
                req.set_input_tensor(index, tensor);
            };
            auto ints = [&](int index, ov::Shape shape, const std::vector<int> &values) {
                ov::Tensor tensor(indices, shape);
                for (size_t i = 0; i < values.size(); ++i) {
                    if (indices == ov::element::i32)
                        tensor.data<int32_t>()[i] = values[i];
                    else
                        tensor.data<int64_t>()[i] = values[i];
                }
                req.set_input_tensor(index, tensor);
            };
            data(0, {1, 1, S, F}, cache);
            ints(1, {1, 1, 1, size_t(B)}, a);
            auto full = a;
            full.insert(full.end(), e.begin(), e.end());
            ints(2, {1, 1, 1, full.size()}, full);
            ints(3, {1, 1, 1, 1}, {rb});
            ints(4, {1, 1, 1, 1}, {rl});
            ints(5, {1, 1, 1, 2}, {eb, B});
            if (conv) {
                data(6, {1, size_t(B), C, size_t(T)}, qkv);
                std::vector<int> w;
                for (auto [start, dst] : windows) {
                    w.push_back(start);
                    w.push_back(dst);
                }
                ints(7, {1, 1, windows.size(), 2}, w);
            }
            req.infer();
            const std::vector<std::vector<float>> expected{expected_active, expected_cache};
            for (size_t output = 0; output < expected.size(); ++output) {
                auto actual = req.get_output_tensor(output);
                ASSERT_EQ(actual.get_size(), expected[output].size());
                for (size_t i = 0; i < actual.get_size(); ++i) {
                    const float value =
                        precision == ov::element::f16 ? float(actual.data<ov::float16>()[i]) : actual.data<float>()[i];
                    ASSERT_EQ(value, expected[output][i])
                        << "step=" << step << " output=" << output << " element=" << i;
                }
            }
            // A kernel must never modify its input, even for cyclic reorders.
            auto original = req.get_input_tensor(0);
            for (size_t i = 0; i < cache.size(); ++i) {
                const float value =
                    precision == ov::element::f16 ? float(original.data<ov::float16>()[i]) : original.data<float>()[i];
                ASSERT_EQ(value, cache[i]);
            }
        }
    }
}
TEST(RecurrentCacheNative, GatherNativeMatrixLayout) {
    for (auto type : {ov::element::f16, ov::element::f32})
        for (auto indices : {ov::element::i32, ov::element::i64})
            run_cache_case(false, type, indices, false, true);
}
TEST(RecurrentCacheNative, DynamicResetReorderSnapshotsAndExport) {
    for (bool conv : {false, true})
        for (auto type : {ov::element::f16, ov::element::f32})
            for (auto indices : {ov::element::i32, ov::element::i64})
                for (bool fixed_shape : {false, true})
                    run_cache_case(conv, type, indices, fixed_shape);
}
} // namespace

TEST(RecurrentCacheNative, RemoteBufferCycles) {
    constexpr size_t S = 4, F = 524288;
    auto old = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, S, F});
    auto ids = ov::op::v0::Constant::create(ov::element::i32, {1, 1, 1, 4}, {1, 0, 3, 2});
    auto zero = ov::op::v0::Constant::create(ov::element::i32, {1, 1, 1, 1}, {0});
    auto offsets = ov::op::v0::Constant::create(ov::element::i32, {1, 1, 1, 2}, {0, 0});
    auto op =
        std::make_shared<ov::op::internal::RecurrentStateGather>(ov::OutputVector{old, ids, ids, zero, zero, offsets});
    ov::Core core;
    auto context = core.get_default_context("GPU").as<ov::intel_gpu::ocl::ClContext>();
    auto compiled = core.compile_model(
        std::make_shared<ov::Model>(ov::OutputVector{op->output(0), op->output(1)}, ov::ParameterVector{old}), context);
    auto req = compiled.create_infer_request();
    auto remote = context.create_tensor(ov::element::f32, {1, 1, S, F});
    ov::Tensor host(ov::element::f32, {1, 1, S, F});
    req.set_input_tensor(0, remote);
    req.set_output_tensor(1, remote);
    for (size_t step = 0; step < 4; ++step) {
        for (size_t r = 0; r < S; ++r)
            std::fill_n(host.data<float>() + r * F, F, float(r + 1 + step));
        host.copy_to(remote);
        req.infer();
        remote.copy_to(host);
        for (size_t r = 0; r < S; ++r)
            for (size_t f = 0; f < F; ++f)
                ASSERT_EQ(host.data<float>()[r * F + f], float((r ^ 1) + 1 + step));
    }
}

TEST(RecurrentCacheNative, StateUpdateDynamicPlanesAndExport) {
    for (auto precision : {ov::element::f16, ov::element::f32}) {
        for (auto indices : {ov::element::i32, ov::element::i64}) {
            for (bool transpose : {false, true}) {
                auto cache = std::make_shared<ov::op::v0::Parameter>(precision, ov::PartialShape{1, 1, 12, 35});
                auto states = std::make_shared<ov::op::v0::Parameter>(precision, ov::PartialShape{1, 1, -1, 35});
                auto control = std::make_shared<ov::op::v0::Parameter>(indices, ov::PartialShape{1, 1, 1, 4});
                auto op = std::make_shared<ov::op::internal::RecurrentStateUpdate>(
                    ov::OutputVector{cache, states, control}, transpose ? 5 : 0, transpose ? 7 : 0);
                ov::Core core;
                auto model = core.compile_model(
                    std::make_shared<ov::Model>(ov::OutputVector{op}, ov::ParameterVector{cache, states, control}),
                    "GPU");
                ASSERT_EQ(op->clone_with_new_inputs({cache, states, control})->get_output_partial_shape(0),
                          cache->get_partial_shape());
                std::stringstream blob;
                model.export_model(blob);
                auto imported = core.import_model(blob, "GPU");
                for (auto compiled : {model, imported}) {
                    auto req = compiled.create_infer_request();
                    for (int step = 0; step < 8; ++step) {
                        const int B = step % 3 + 1, planes = step % 3 + 1, start = step % 2;
                        const int stride = step % 2 ? B : 1, begin = step % 3;
                        const int N = B * planes + start;
                        ov::Tensor a(precision, {1, 1, 12, 35}), b(precision, {1, 1, size_t(N), 35});
                        auto set = [&](ov::Tensor &t, size_t i, float v) {
                            if (precision == ov::element::f16)
                                t.data<ov::float16>()[i] = ov::float16(v);
                            else
                                t.data<float>()[i] = v;
                        };
                        auto get = [&](const ov::Tensor &t, size_t i) {
                            return precision == ov::element::f16 ? float(t.data<ov::float16>()[i]) : t.data<float>()[i];
                        };
                        std::vector<float> expected(a.get_size());
                        for (size_t i = 0; i < a.get_size(); ++i) {
                            set(a, i, float(i % 19));
                            expected[i] = get(a, i);
                        }
                        for (size_t i = 0; i < b.get_size(); ++i)
                            set(b, i, float(i % 31 + 32));
                        for (int r = 0; r < N - start; ++r)
                            for (int f = 0; f < 35; ++f)
                                expected[(begin + (r / B) * stride + r % B) * 35 + f] =
                                    get(b, (start + r) * 35 + (transpose ? (f % 5) * 7 + f / 5 : f));
                        ov::Tensor c(indices, {1, 1, 1, 4});
                        const int values[] = {begin, B, stride, start};
                        for (int i = 0; i < 4; ++i) {
                            if (indices == ov::element::i32)
                                c.data<int32_t>()[i] = values[i];
                            else
                                c.data<int64_t>()[i] = values[i];
                        }
                        req.set_input_tensor(0, a);
                        req.set_input_tensor(1, b);
                        req.set_input_tensor(2, c);
                        req.infer();
                        auto actual = req.get_output_tensor(0);
                        ASSERT_EQ(actual.get_size(), expected.size());
                        for (size_t i = 0; i < expected.size(); ++i) {
                            ASSERT_EQ(get(actual, i), expected[i]) << "step=" << step << " i=" << i;
                            ASSERT_EQ(get(a, i), float(i % 19));
                        }
                    }
                }
            }
        }
    }
}
