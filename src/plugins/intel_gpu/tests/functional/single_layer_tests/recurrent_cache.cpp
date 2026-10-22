// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <openvino/core/type/float16.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gated_delta_net.hpp>
#include <openvino/op/group_conv.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/minimum.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/recurrent_cache.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#include <openvino/runtime/properties.hpp>
#include <sstream>
#include <utility>
#include <vector>

namespace {
// Run changing B/T/maps/reset offsets through the same compiled graph and imported blob.
// Reference deliberately uses a reset copy, followed by gather and ordered row writes.
void run_cache_case(bool conv,
                    ov::element::Type precision,
                    ov::element::Type indices,
                    bool fixed_shape = false,
                    bool transpose = false,
                    ov::element::Type cache_precision = ov::element::dynamic) {
    if (cache_precision == ov::element::dynamic)
        cache_precision = precision;
    constexpr int S = 12, C = 7, H = 3, F = C * H;
    ov::ParameterVector params;
    ov::OutputVector args;
    auto add = [&](ov::element::Type type, ov::PartialShape shape) {
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        params.push_back(param);
        args.push_back(param);
    };
    add(cache_precision, {1, 1, S, F});
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
    // Preserve declared F16/F32 types: a global inference hint would mask mixed-precision writes.
    auto compiled = core.compile_model(std::make_shared<ov::Model>(results, params), "GPU", ov::hint::inference_precision(ov::element::dynamic));
    auto clone = op->clone_with_new_inputs(args);
    ASSERT_EQ(clone->get_output_partial_shape(0), op->get_output_partial_shape(0));
    std::stringstream blob;
    compiled.export_model(blob);
    auto imported = core.import_model(blob, "GPU", ov::hint::inference_precision(ov::element::dynamic));
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
                for (int k = 0; k < (step == 7 ? 0 : 3); ++k)
                    windows.emplace_back(std::max(0, T - 2 + k), 8 - 4 * k);
            std::vector<float> cache(S * F), qkv(B * C * T);
            auto round = [&](float x) {
                return precision == ov::element::f16 ? float(ov::float16(x)) : x;
            };
            auto round_cache = [&](float x) {
                return cache_precision == ov::element::f16 ? float(ov::float16(x)) : x;
            };
            for (size_t i = 0; i < cache.size(); ++i)
                cache[i] = round_cache(float(int(i % 89) - 40) / 7);
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
                            expected_active[(b * C + c) * (H + T) + t] = t < H ? round(cleared[a[b] * F + c * H + t]) : qkv[(b * C + c) * T + t - H];
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
                            expected_cache[(dst + b) * F + c * H + t] = round_cache(expected_active[(b * C + c) * (H + T) + start + t]);
            auto data = [&](int index, ov::Shape shape, const std::vector<float>& values) {
                const auto type = index == 0 ? cache_precision : precision;
                ov::Tensor tensor(type, shape);
                for (size_t i = 0; i < values.size(); ++i) {
                    if (type == ov::element::f16)
                        tensor.data<ov::float16>()[i] = ov::float16(values[i]);
                    else
                        tensor.data<float>()[i] = values[i];
                }
                req.set_input_tensor(index, tensor);
            };
            auto ints = [&](int index, ov::Shape shape, const std::vector<int>& values) {
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
                    const float value = actual.get_element_type() == ov::element::f16 ? float(actual.data<ov::float16>()[i]) : actual.data<float>()[i];
                    ASSERT_EQ(value, expected[output][i]) << "step=" << step << " output=" << output << " element=" << i;
                }
            }
            // A kernel must never modify its input, even for cyclic reorders.
            auto original = req.get_input_tensor(0);
            for (size_t i = 0; i < cache.size(); ++i) {
                const float value = cache_precision == ov::element::f16 ? float(original.data<ov::float16>()[i]) : original.data<float>()[i];
                ASSERT_EQ(value, cache[i]);
            }
        }
    }
}
TEST(RecurrentCacheNative, MixedPrecisionConvolutionWriteback) {
    for (auto type : {ov::element::f16, ov::element::f32})
        for (auto indices : {ov::element::i32, ov::element::i64})
            run_cache_case(true, type, indices, false, false, type == ov::element::f16 ? ov::element::f32 : ov::element::f16);
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
}  // namespace

TEST(RecurrentCacheNative, RemoteBufferCycles) {
    constexpr size_t S = 4, F = 524288;
    auto old = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, S, F});
    auto ids = ov::op::v0::Constant::create(ov::element::i32, {1, 1, 1, 4}, {1, 0, 3, 2});
    auto zero = ov::op::v0::Constant::create(ov::element::i32, {1, 1, 1, 1}, {0});
    auto offsets = ov::op::v0::Constant::create(ov::element::i32, {1, 1, 1, 2}, {0, 0});
    auto op = std::make_shared<ov::op::internal::RecurrentStateGather>(ov::OutputVector{old, ids, ids, zero, zero, offsets});
    ov::Core core;
    auto context = core.get_default_context("GPU").as<ov::intel_gpu::ocl::ClContext>();
    auto compiled = core.compile_model(std::make_shared<ov::Model>(ov::OutputVector{op->output(0), op->output(1)}, ov::ParameterVector{old}), context);
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
                auto op =
                    std::make_shared<ov::op::internal::RecurrentStateUpdate>(ov::OutputVector{cache, states, control}, transpose ? 5 : 0, transpose ? 7 : 0);
                ov::Core core;
                auto model = core.compile_model(std::make_shared<ov::Model>(ov::OutputVector{op}, ov::ParameterVector{cache, states, control}),
                                                "GPU",
                                                ov::hint::inference_precision(ov::element::dynamic));
                ASSERT_EQ(op->clone_with_new_inputs({cache, states, control})->get_output_partial_shape(0), cache->get_partial_shape());
                std::stringstream blob;
                model.export_model(blob);
                auto imported = core.import_model(blob, "GPU", ov::hint::inference_precision(ov::element::dynamic));
                for (auto compiled : {model, imported}) {
                    auto req = compiled.create_infer_request();
                    for (int step = 0; step < 8; ++step) {
                        const int B = step % 3 + 1, planes = step % 3 + 1, start = step % 2;
                        const int stride = step % 3 == 0 ? 0 : step % 2 ? B : 1, begin = step % 3;
                        const int N = step == 7 ? start : B * planes + start - step % B;
                        ov::Tensor a(precision, {1, 1, 12, 35}), b(precision, {1, 1, size_t(N), 35});
                        auto set = [&](ov::Tensor& t, size_t i, float v) {
                            if (precision == ov::element::f16)
                                t.data<ov::float16>()[i] = ov::float16(v);
                            else
                                t.data<float>()[i] = v;
                        };
                        auto get = [&](const ov::Tensor& t, size_t i) {
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
                                expected[(begin + (r / B) * stride + r % B) * 35 + f] = get(b, (start + r) * 35 + (transpose ? (f % 5) * 7 + f / 5 : f));
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

// Compare deferred reset/reorder with the original two-output gather followed by update.
// The same dynamic graph and its exported blob must handle cycles, duplicate sources,
// reset-before-gather, empty reorder suffixes and overlapping snapshot destinations.
TEST(RecurrentCacheNative, DeferredPreparationMatchesMaterializedCache) {
    for (auto precision : {ov::element::f16, ov::element::f32}) {
        for (auto indices : {ov::element::i32, ov::element::i64}) {
            constexpr size_t S = 12, F = 35;
            ov::ParameterVector params;
            auto param = [&](ov::element::Type type, ov::PartialShape shape) {
                auto p = std::make_shared<ov::op::v0::Parameter>(type, shape);
                params.push_back(p);
                return p;
            };
            auto cache = param(precision, {1, 1, S, F});
            auto states = param(precision, {1, 1, -1, F});
            auto control = param(indices, {1, 1, 1, 4});
            auto active_rows = param(indices, {1, 1, 1, -1});
            auto extra_rows = param(indices, {1, 1, 1, -1});
            auto rb = param(indices, {1, 1, 1, 1});
            auto rl = param(indices, {1, 1, 1, 1});
            auto offsets = param(indices, {1, 1, 1, 2});
            ov::OutputVector gather_args{cache, active_rows, extra_rows, rb, rl, offsets};
            auto original = std::make_shared<ov::op::internal::RecurrentStateGather>(gather_args, 5, 7);
            auto active = std::make_shared<ov::op::internal::RecurrentStateGather>(gather_args, 5, 7, false);
            ASSERT_EQ(active->get_output_size(), 1);
            auto reference = std::make_shared<ov::op::internal::RecurrentStateUpdate>(ov::OutputVector{original->output(1), states, control}, 5, 7);
            auto fused = std::make_shared<ov::op::internal::RecurrentStateUpdate>(ov::OutputVector{cache, states, control, extra_rows, rb, rl, offsets}, 5, 7);
            ov::Core core;
            auto graph = std::make_shared<ov::Model>(ov::OutputVector{active->output(0), original->output(0), fused->output(0), reference->output(0)}, params);
            auto compiled = core.compile_model(graph, "GPU", ov::hint::inference_precision(ov::element::dynamic));
            std::stringstream blob;
            compiled.export_model(blob);
            auto imported = core.import_model(blob, "GPU", ov::hint::inference_precision(ov::element::dynamic));
            for (auto model : {compiled, imported}) {
                auto request = model.create_infer_request();
                for (int step = 0; step < 8; ++step) {
                    const int B = 1 + step % 3, N = step == 7 ? 0 : 3 * B;
                    auto set_indices = [&](size_t port, std::vector<int> values) {
                        ov::Tensor tensor(indices, {1, 1, 1, values.size()});
                        for (size_t i = 0; i < values.size(); ++i) {
                            if (indices == ov::element::i32)
                                tensor.data<int32_t>()[i] = values[i];
                            else
                                tensor.data<int64_t>()[i] = values[i];
                        }
                        request.set_input_tensor(port, tensor);
                    };
                    auto set_data = [&](size_t port, size_t rows) {
                        ov::Tensor tensor(precision, {1, 1, rows, F});
                        for (size_t i = 0; i < tensor.get_size(); ++i) {
                            const float value = float(int((i + step + port * 29) % 97) - 48) / 7;
                            if (precision == ov::element::f16)
                                tensor.data<ov::float16>()[i] = ov::float16(value);
                            else
                                tensor.data<float>()[i] = value;
                        }
                        request.set_input_tensor(port, tensor);
                    };
                    set_data(0, S);
                    set_data(1, N);
                    set_indices(2, {0, B, step % 2 ? B : 0, 0});
                    std::vector<int> a(B, 1);
                    if (B > 1)
                        a[1] = 0;
                    set_indices(3, a);
                    set_indices(4, {1, 0, 1});
                    set_indices(5, {step % 3});
                    set_indices(6, {step % 2});
                    set_indices(7, {0, step % 2 ? 3 : 0});
                    request.infer();
                    for (size_t port : {0u, 2u}) {
                        auto actual = request.get_output_tensor(port), expected = request.get_output_tensor(port + 1);
                        ASSERT_EQ(actual.get_shape(), expected.get_shape());
                        for (size_t i = 0; i < actual.get_size(); ++i) {
                            if (precision == ov::element::f16)
                                ASSERT_EQ(float(actual.data<ov::float16>()[i]), float(expected.data<ov::float16>()[i]));
                            else
                                ASSERT_EQ(actual.data<float>()[i], expected.data<float>()[i]);
                        }
                    }
                }
            }
        }
    }
}

TEST(RecurrentCacheNative, FusedGDNResetReorderSnapshots) {
    constexpr int S = 12, H = 2;
    for (const auto& dimensions : {std::pair<int, int>{128, 8}, std::pair<int, int>{8, 7}}) {
        const int K = dimensions.first, V = dimensions.second, F = H * K * V;
        for (int count : {1, 3}) {
            for (bool fp16 : {false, true}) {
                ov::ParameterVector params;
                auto param = [&](ov::element::Type type, ov::PartialShape shape) {
                    auto p = std::make_shared<ov::op::v0::Parameter>(type, shape);
                    params.push_back(p);
                    return p;
                };
                const auto type = fp16 ? ov::element::f16 : ov::element::f32;
                auto q = param(type, {-1, -1, H, K}), k = param(type, {-1, -1, H, K}), v = param(type, {-1, -1, H, V});
                auto cache = param(ov::element::f32, {1, 1, -1, F});
                auto g = param(type, {-1, -1, H}), beta = param(type, {-1, -1, H});
                auto active = param(ov::element::i32, {1, 1, 1, -1}), extra = param(ov::element::i32, {1, 1, 1, -1});
                auto rb = param(ov::element::i32, {1, 1, 1, 1}), rl = param(ov::element::i32, {1, 1, 1, 1});
                auto off = param(ov::element::i32, {1, 1, 1, 2}), control = param(ov::element::i32, {1, 1, 1, 4});
                auto rounded = std::make_shared<ov::op::v0::Convert>(cache, type);
                auto gather = std::make_shared<ov::op::internal::RecurrentStateGather>(ov::OutputVector{rounded, active, extra, rb, rl, off}, K, V);
                auto pattern = ov::op::v0::Constant::create(ov::element::i64, {4}, {-1, H, K, V});
                auto state = std::make_shared<ov::op::v1::Reshape>(gather->output(0), pattern, false);
                auto gdn = std::make_shared<ov::op::internal::GatedDeltaNet>(ov::OutputVector{q, k, v, state, g, beta}, true, 1e-6f, 1e-6f, count);
                ov::Output<ov::Node> snapshots = gdn->output(1);
                if (count > 1) {
                    // Only initialized newest-first snapshots are written for short batches.
                    auto tokens = std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::ShapeOf>(q),
                                                                       ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                                                       ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
                    auto valid = std::make_shared<ov::op::v1::Minimum>(tokens, ov::op::v0::Constant::create(ov::element::i64, {1}, {count}));
                    snapshots = std::make_shared<ov::op::v8::Slice>(snapshots,
                                                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                                                                    valid,
                                                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
                }
                auto flatten = std::make_shared<ov::op::v1::Reshape>(snapshots, ov::op::v0::Constant::create(ov::element::i64, {4}, {1, 1, -1, F}), false);
                auto update = std::make_shared<ov::op::internal::RecurrentStateUpdate>(ov::OutputVector{gather->output(1), flatten, control}, K, V);
                auto reference = std::make_shared<ov::op::v0::Convert>(update, ov::element::f32);
                auto fused =
                    std::make_shared<ov::op::internal::RecurrentGatedDeltaNet>(ov::OutputVector{q, k, v, cache, g, beta, active, extra, rb, rl, off, control},
                                                                               true,
                                                                               1e-6f,
                                                                               1e-6f,
                                                                               count);
                ov::Core core;
                auto compiled =
                    core.compile_model(std::make_shared<ov::Model>(ov::OutputVector{fused->output(0), gdn->output(0), fused->output(1), reference}, params),
                                       "GPU",
                                       ov::hint::inference_precision(ov::element::dynamic));
                std::stringstream blob;
                compiled.export_model(blob);
                auto imported = core.import_model(blob, "GPU", ov::hint::inference_precision(ov::element::dynamic));
                for (auto model : {compiled, imported}) {
                    auto req = model.create_infer_request();
                    for (int step = 0; step < 6; ++step) {
                        const size_t B = step % 3 + 1, T = step == 5 ? 17 : step % 4 + 1;
                        std::vector<ov::Shape> shapes{{B, T, H, size_t(K)},
                                                      {B, T, H, size_t(K)},
                                                      {B, T, H, size_t(V)},
                                                      {1, 1, S, size_t(F)},
                                                      {B, T, H},
                                                      {B, T, H}};
                        for (size_t port = 0; port < 6; ++port) {
                            auto tp = port == 3 ? ov::element::f32 : type;
                            ov::Tensor tensor(tp, shapes[port]);
                            for (size_t i = 0; i < tensor.get_size(); ++i) {
                                float x = port == 4 ? -0.3f : port == 5 ? 0.2f : float(int((i * 7 + step) % 53) - 26) / 100;
                                if (tp == ov::element::f16)
                                    tensor.data<ov::float16>()[i] = ov::float16(x);
                                else
                                    tensor.data<float>()[i] = x;
                            }
                            if (port == 3) {
                                // Alternate direct distinct outputs and aliased fallback on the same
                                // request, including export/import and changing B/T. Aliased maps
                                // need immutable reads; distinct outputs must leave input unchanged.
                                auto context = core.get_default_context("GPU").as<ov::intel_gpu::ocl::ClContext>();
                                auto remote = context.create_usm_device_tensor(ov::element::f32, tensor.get_shape());
                                tensor.copy_to(remote);
                                req.set_input_tensor(port, remote);
                                req.set_output_tensor(2, step % 3 == 1 ? remote :
                                    context.create_usm_device_tensor(ov::element::f32, tensor.get_shape()));
                            } else {
                                req.set_input_tensor(port, tensor);
                            }
                        }
                        auto ints = [&](int port, std::vector<int> values) {
                            ov::Tensor t(ov::element::i32, {1, 1, 1, values.size()});
                            std::copy(values.begin(), values.end(), t.data<int>());
                            req.set_input_tensor(port, t);
                        };
                        std::vector<int> a(B, 1);
                        if (B > 1)
                            a[1] = 0;
                        ints(6, a);
                        ints(7, {1, 0, 1});
                        ints(8, {step % 3});
                        ints(9, {step % 2});
                        ints(10, {0, step % 2 ? 3 : 0});
                        ints(11, {0, int(B), step % 2 ? int(B) : 0, step % 2});
                        req.infer();
                        if (step % 3 != 1) {
                            auto input = req.get_input_tensor(3);
                            ov::Tensor host(ov::element::f32, input.get_shape());
                            input.copy_to(host);
                            for (size_t i = 0; i < host.get_size(); ++i)
                                ASSERT_EQ(host.data<float>()[i], float(int((i * 7 + step) % 53) - 26) / 100);
                        }
                        for (size_t port : {0u, 2u}) {
                            auto actual = req.get_output_tensor(port), expected = req.get_output_tensor(port + 1);
                            if (port == 2) {
                                ov::Tensor host(ov::element::f32, actual.get_shape());
                                actual.copy_to(host);
                                actual = host;
                            }
                            ASSERT_EQ(actual.get_shape(), expected.get_shape());
                            for (size_t i = 0; i < actual.get_size(); ++i) {
                                float x = actual.get_element_type() == ov::element::f16 ? float(actual.data<ov::float16>()[i]) : actual.data<float>()[i];
                                float y = expected.get_element_type() == ov::element::f16 ? float(expected.data<ov::float16>()[i]) : expected.data<float>()[i];
                                ASSERT_EQ(x, y) << "step=" << step << " port=" << port << " i=" << i;
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST(RecurrentCacheNative, FusedConvolutionResetReorderWindows) {
    constexpr size_t S = 12, C = 7, H = 3, F = C * H;
    for (auto type : {ov::element::f16, ov::element::f32}) {
        ov::ParameterVector params;
        auto param = [&](ov::element::Type t, ov::PartialShape shape) {
            auto p = std::make_shared<ov::op::v0::Parameter>(t, shape);
            params.push_back(p);
            return p;
        };
        auto cache = param(ov::element::f32, {1, 1, -1, F}), a = param(ov::element::i32, {1, 1, 1, -1}), e = param(ov::element::i32, {1, 1, 1, -1});
        auto rb = param(ov::element::i32, {1, 1, 1, 1}), rl = param(ov::element::i32, {1, 1, 1, 1}), off = param(ov::element::i32, {1, 1, 1, 2});
        auto qkv = param(type, {1, -1, C, -1}), w = param(ov::element::i32, {1, 1, -1, 2}), weights = param(type, {1, 1, C, H + 1});
        ov::OutputVector args{cache, a, e, rb, rl, off, qkv, w, weights};
        auto fused = std::make_shared<ov::op::internal::RecurrentConvCache>(args, H);
        ov::OutputVector refargs{std::make_shared<ov::op::v0::Convert>(cache, type), a, e, rb, rl, off, qkv, w};
        auto prepared = std::make_shared<ov::op::internal::RecurrentConvCache>(refargs, H);
        auto x = std::make_shared<ov::op::v0::Squeeze>(prepared->output(0), ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        auto filter =
            std::make_shared<ov::op::v1::Reshape>(weights, ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{C, 1, 1, H + 1}), false);
        auto conv = std::make_shared<ov::op::v1::GroupConvolution>(x, filter, ov::Strides{1}, ov::CoordinateDiff{0}, ov::CoordinateDiff{0}, ov::Strides{1});
        auto refcache = std::make_shared<ov::op::v0::Convert>(prepared->output(1), ov::element::f32);
        // The cache Result can overlap a different input with an independent reader.
        // Direct binding must reject it even though it is not a kernel dependency.
        auto unrelated = param(ov::element::f32, {1, 1, -1, F});
        auto branch = std::make_shared<ov::op::v1::Multiply>(unrelated, unrelated);
        ov::Core core;
        auto compiled = core.compile_model(std::make_shared<ov::Model>(ov::OutputVector{fused->output(0), conv, fused->output(1), refcache, branch}, params),
                                           "GPU",
                                           ov::hint::inference_precision(ov::element::dynamic));
        std::stringstream blob;
        compiled.export_model(blob);
        auto imported = core.import_model(blob, "GPU", ov::hint::inference_precision(ov::element::dynamic));
        for (auto model : {compiled, imported}) {
            auto req = model.create_infer_request();
            for (size_t step = 0; step < 8; ++step) {
                size_t B = step % 3 + 1, T = step == 5 ? 16 : step == 6 ? 33 : step % 4 + 1;
                auto data = [&](int port, ov::Shape shape) {
                    auto tp = port == 0 ? ov::element::f32 : type;
                    ov::Tensor t(tp, shape);
                    for (size_t i = 0; i < t.get_size(); ++i) {
                        float x = float(int((i * 7 + (port == 8 ? 0 : step)) % 53) - 26) / 32;
                        if (tp == ov::element::f16)
                            t.data<ov::float16>()[i] = ov::float16(x);
                        else
                            t.data<float>()[i] = x;
                    }
                    if (port == 0) {
                        auto context = core.get_default_context("GPU").as<ov::intel_gpu::ocl::ClContext>();
                        auto remote = context.create_usm_device_tensor(ov::element::f32, shape);
                        t.copy_to(remote);
                        req.set_input_tensor(port, remote);
                        req.set_output_tensor(2, step % 3 == 1 ? remote :
                            context.create_usm_device_tensor(ov::element::f32, shape));
                    } else {
                        req.set_input_tensor(port, t);
                    }
                };
                auto ints = [&](int port, ov::Shape shape, std::vector<int> vals) {
                    ov::Tensor t(ov::element::i32, shape);
                    std::copy(vals.begin(), vals.end(), t.data<int>());
                    req.set_input_tensor(port, t);
                };
                data(0, {1, 1, S, F});
                data(6, {1, B, C, T});
                data(8, {1, 1, C, H + 1});
                auto context = core.get_default_context("GPU").as<ov::intel_gpu::ocl::ClContext>();
                ov::Tensor branch_input(ov::element::f32, {1, 1, S, F});
                for (size_t i = 0; i < branch_input.get_size(); ++i)
                    branch_input.data<float>()[i] = float(i % 11 + 1);
                auto branch_remote = context.create_usm_device_tensor(ov::element::f32, branch_input.get_shape());
                branch_input.copy_to(branch_remote);
                req.set_input_tensor(9, branch_remote);
                if (step % 3 == 2)
                    req.set_output_tensor(2, branch_remote);
                std::vector<int> ids(B, 1);
                if (B > 1)
                    ids[1] = 0;
                ints(1, {1, 1, 1, B}, ids);
                ints(2, {1, 1, 1, 3}, {1, 0, 1});
                ints(3, {1, 1, 1, 1}, {int(step % 3)});
                ints(4, {1, 1, 1, 1}, {int(step % 2)});
                ints(5, {1, 1, 1, 2}, {0, step % 2 ? 3 : 0});
                std::vector<int> windows = step == 7 ? std::vector<int>{} : std::vector<int>{0, 0, int(T), 0, int(T / 2), 4};
                ints(7, {1, 1, windows.size() / 2, 2}, windows);
                req.infer();
                auto out = req.get_output_tensor(0), ref = req.get_output_tensor(1);
                for (size_t b = 0; b < B; ++b)
                    for (size_t t = 0; t < T; ++t)
                        for (size_t c = 0; c < C; ++c) {
                            size_t i = (b * T + t) * C + c, j = (b * C + c) * T + t;
                            float x = type == ov::element::f16 ? float(out.data<ov::float16>()[i]) : out.data<float>()[i];
                            float y = type == ov::element::f16 ? float(ref.data<ov::float16>()[j]) : ref.data<float>()[j];
                            ASSERT_NEAR(x, y, type == ov::element::f16 ? 0.002f : 1e-6f)
                                << "step=" << step << " b=" << b << " t=" << t << " c=" << c << " type=" << type;
                        }
                auto branch_result = req.get_output_tensor(4);
                for (size_t i = 0; i < branch_result.get_size(); ++i) {
                    float value = branch_input.data<float>()[i];
                    ASSERT_EQ(branch_result.data<float>()[i], value * value);
                }
                out = req.get_output_tensor(2);
                ov::Tensor host(ov::element::f32, out.get_shape());
                out.copy_to(host);
                out = host;
                if (step % 3 != 1) {
                    auto input = req.get_input_tensor(0);
                    ov::Tensor initial(ov::element::f32, input.get_shape());
                    input.copy_to(initial);
                    for (size_t i = 0; i < initial.get_size(); ++i)
                        ASSERT_EQ(initial.data<float>()[i], float(int((i * 7 + step) % 53) - 26) / 32);
                }
                ref = req.get_output_tensor(3);
                for (size_t i = 0; i < S * F; ++i)
                    ASSERT_EQ(out.data<float>()[i], ref.data<float>()[i]);
            }
        }
    }
}
