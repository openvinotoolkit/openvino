// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/op/recurrent_cache_update.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <sstream>
#include <vector>

#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/gated_delta_net.hpp"
#include <cmath>
#include "openvino/op/reshape.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"
#include "openvino/runtime/properties.hpp"

namespace {
float get_value(const ov::Tensor& t, size_t i) {
    return t.get_element_type() == ov::element::f16 ? float(t.data<ov::float16>()[i]) : t.data<float>()[i];
}
void set_value(ov::Tensor& t, size_t i, float value) {
    if (t.get_element_type() == ov::element::f16)
        t.data<ov::float16>()[i] = ov::float16(value);
    else
        t.data<float>()[i] = value;
}
float round_value(float value, ov::element::Type type) {
    return type == ov::element::f16 ? float(ov::float16(value)) : value;
}
// Arithmetic is checked against a host convolution or ordinary GDN. Reset,
// gather, reorder and ordered cache writes are calculated independently on host.
void run_update_case(bool conv,
                     ov::element::Type type,
                     ov::element::Type indices,
                     int K,
                     int V,
                     int snapshots,
                     bool other_reader = false,
                     bool unrelated_overlap = false,
                     ov::element::Type cache_type = ov::element::f32,
                     bool cache_view = false,
                     bool host_usm = false) {
    constexpr int S = 12, H = 2, C = 7, history = 3;
    const int F = conv ? C * history : H * K * V;
    const size_t cache_port = conv ? 0 : 3;
    ov::ParameterVector params;
    ov::OutputVector args;
    auto add = [&](ov::element::Type t, ov::PartialShape shape) {
        auto p = std::make_shared<ov::op::v0::Parameter>(t, shape);
        params.push_back(p);
        args.push_back(p);
    };
    if (conv) {
        add(cache_type, {1, 1, -1, F});
        for (int i = 0; i < 2; ++i)
            add(indices, {1, 1, 1, -1});
        for (int i = 0; i < 2; ++i)
            add(indices, {1, 1, 1, 1});
        add(indices, {1, 1, 1, 2});
        add(type, {1, -1, C, -1});
        add(indices, {1, 1, -1, 2});
        add(type, {1, 1, C, history + 1});
    } else {
        add(type, {-1, -1, H, K});
        add(type, {-1, -1, H, K});
        add(type, {-1, -1, H, V});
        add(cache_type, {1, 1, -1, F});
        for (int i = 0; i < 2; ++i)
            add(type, {-1, -1, H});
        for (int i = 0; i < 2; ++i)
            add(indices, {1, 1, 1, -1});
        for (int i = 0; i < 2; ++i)
            add(indices, {1, 1, 1, 1});
        add(indices, {1, 1, 1, 2});
        add(indices, {1, 1, 1, 4});
    }
    if (cache_view) {
        auto pattern = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, 1, -1, F});
        args[cache_port] = std::make_shared<ov::op::v1::Reshape>(params[cache_port], pattern, false);
    }
    std::shared_ptr<ov::Node> update;
    ov::Core core;
    const auto hint = ov::hint::inference_precision(ov::element::dynamic);
    ov::CompiledModel ref;
    if (conv) {
        update = std::make_shared<ov::op::internal::RecurrentConvCacheUpdate>(args, history);
    } else {
        update = std::make_shared<ov::op::internal::RecurrentGatedDeltaNetUpdate>(args, true, 1e-6f, 1e-6f, snapshots);
        auto state = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape{-1, H, K, V});
        auto gdn = std::make_shared<ov::op::internal::GatedDeltaNet>(
            ov::OutputVector{args[0], args[1], args[2], state, args[4], args[5]}, true, 1e-6f, 1e-6f);
        ref = core.compile_model(std::make_shared<ov::Model>(gdn->outputs(),
            ov::ParameterVector{params[0], params[1], params[2], state, params[4], params[5]}), "GPU", hint);
    }
    ov::OutputVector outputs{update->output(0), update->output(1)};
    if (other_reader)
        outputs.push_back(params[cache_port]);
    std::shared_ptr<ov::op::v0::Parameter> unrelated;
    if (unrelated_overlap) {
        unrelated = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, F});
        params.push_back(unrelated);
        outputs.push_back(std::make_shared<ov::op::v1::Multiply>(unrelated, unrelated));
    }
    auto compiled = core.compile_model(std::make_shared<ov::Model>(outputs, params), "GPU", hint);
    std::stringstream blob;
    compiled.export_model(blob);
    auto imported = core.import_model(blob, "GPU", hint);
    auto context = core.get_default_context("GPU").as<ov::intel_gpu::ocl::ClContext>();
    for (auto model : {compiled, imported}) {
        auto req = model.create_infer_request();
        ov::InferRequest expected_req;
        if (!conv)
            expected_req = ref.create_infer_request();
        for (int step = 0; step < 9; ++step) {
            const size_t B = step % 4 + 1, T = step == 8 ? 17 : step % 4 + 1;
            std::vector<ov::Tensor> inputs;
            auto data = [&](size_t port, ov::Shape shape) {
                ov::Tensor t(params[port]->get_element_type(), shape);
                for (size_t i = 0; i < t.get_size(); ++i) {
                    float x = !conv && port == 4 ? -0.3f : !conv && port == 5 ? 0.2f : float(int((i * 7 + step) % 53) - 26) / 100;
                    if (t.get_element_type() == ov::element::f16)
                        t.data<ov::float16>()[i] = ov::float16(x);
                    else
                        t.data<float>()[i] = x;
                }
                inputs.push_back(t);
            };
            auto ints = [&](ov::Shape shape, std::vector<int> values) {
                ov::Tensor t(indices, shape);
                for (size_t i = 0; i < values.size(); ++i) {
                    if (indices == ov::element::i32)
                        t.data<int32_t>()[i] = values[i];
                    else
                        t.data<int64_t>()[i] = values[i];
                }
                inputs.push_back(t);
            };
            std::vector<int> active(B, 1);
            if (B > 1)
                active[1] = 0;
            // Reset-before-gather, a cyclic suffix, and duplicate sources.
            std::vector<int> extra = step == 7 ? std::vector<int>{} : std::vector<int>{1, 0, 1};
            // An identity suffix must remain valid without staging its sources,
            // including rows also consumed by active recurrence/snapshot writers.
            if (step == 6)
                extra = {0, 1, 2, 3};
            if (step % 2)
                extra.insert(extra.begin(), active.begin(), active.end());
            const size_t suffix = step % 2 ? B : 0;
            if (conv) {
                data(0, {1, 1, S, size_t(F)});
                ints({1, 1, 1, B}, active);
                ints({1, 1, 1, extra.size()}, extra);
                ints({1, 1, 1, 1}, {step % 3});
                ints({1, 1, 1, 1}, {step % 2});
                ints({1, 1, 1, 2}, {0, int(suffix)});
                data(6, {1, B, C, T});
                // Later windows overwrite earlier windows at overlapping destinations.
                const std::vector<int> windows = step == 7 ? std::vector<int>{} : std::vector<int>{0, 0, int(T), 4, int(T), 2};
                ints({1, 1, windows.size() / 2, 2}, windows);
                data(8, {1, 1, C, history + 1});
            } else {
                data(0, {B, T, H, size_t(K)});
                data(1, {B, T, H, size_t(K)});
                data(2, {B, T, H, size_t(V)});
                data(3, {1, 1, S, size_t(F)});
                data(4, {B, T, H});
                data(5, {B, T, H});
                ints({1, 1, 1, B}, active);
                ints({1, 1, 1, extra.size()}, extra);
                ints({1, 1, 1, 1}, {step % 3});
                ints({1, 1, 1, 1}, {step % 2});
                ints({1, 1, 1, 2}, {0, int(suffix)});
                ints({1, 1, 1, 4}, {0, int(B), step % 2 ? int(B) : 0, step % 2});
            }
            auto integer = [&](size_t port, size_t i = 0) {
                return indices == ov::element::i32 ? int(inputs[port].data<int32_t>()[i]) : int(inputs[port].data<int64_t>()[i]);
            };
            const int rb = integer(conv ? 3 : 8), re = rb + integer(conv ? 4 : 9);
            auto old = [&](int row, int f) {
                return row >= rb && row < re ? 0.f : round_value(get_value(inputs[cache_port], row * F + f), type);
            };
            std::vector<ov::Tensor> expected;
            expected.emplace_back(type, conv ? ov::Shape{1, B, T, C} : ov::Shape{B, T, H, size_t(V)});
            expected.emplace_back(cache_type, ov::Shape{1, 1, S, size_t(F)});
            const int eb = integer(conv ? 5 : 10), es = integer(conv ? 5 : 10, 1);
            for (int row = 0; row < S; ++row) {
                const int src = row >= eb && row < eb + int(extra.size()) - es
                                    ? integer(conv ? 2 : 7, row - eb + es) : row;
                for (int f = 0; f < F; ++f)
                    set_value(expected[1], row * F + f, old(src, f));
            }
            if (conv) {
                auto x = [&](int b, int c, int t) {
                    return t < history ? old(active[b], c * history + t) : get_value(inputs[6], (b * C + c) * T + t - history);
                };
                for (size_t b = 0; b < B; ++b)
                    for (int c = 0; c < C; ++c) {
                        for (size_t t = 0; t < T; ++t) {
                            float value = 0;
                            for (int j = 0; j <= history; ++j)
                                value = std::fma(x(b, c, t + j), get_value(inputs[8], c * (history + 1) + j), value);
                            set_value(expected[0], (b * T + t) * C + c, value);
                        }
                    }
                for (size_t w = 0; w < inputs[7].get_size() / 2; ++w) {
                    const int start = integer(7, 2 * w), dest = integer(7, 2 * w + 1);
                    for (size_t b = 0; b < B; ++b)
                        for (int f = 0; f < F; ++f)
                            set_value(expected[1], (dest + b) * F + f, x(b, f / history, start + f % history));
                }
            } else {
                ov::Tensor state(type, {B, H, size_t(K), size_t(V)});
                for (size_t b = 0; b < B; ++b)
                    for (int h = 0; h < H; ++h)
                        for (int k = 0; k < K; ++k)
                            for (int v = 0; v < V; ++v)
                                set_value(state, ((b * H + h) * K + k) * V + v, old(active[b], (h * V + v) * K + k));
                for (size_t port = 0; port < 6; ++port)
                    expected_req.set_input_tensor(port, port == 3 ? state : inputs[port]);
                expected_req.infer();
                expected_req.get_output_tensor(0).copy_to(expected[0]);
                // Ordinary GDN returns only the final state. Re-run each token prefix
                // from the same initial state to reference newest-first snapshots.
                // Copy each batch separately: shortening [B,T,...] changes its stride.
                const int begin = integer(11), sequences = integer(11, 1), stride = integer(11, 2), source = integer(11, 3);
                for (int slot = 0; slot < std::min<int>(T, snapshots); ++slot) {
                    if (slot > 0) {
                        const size_t prefix = T - slot;
                        for (size_t port : {0, 1, 2, 4, 5}) {
                            auto shape = inputs[port].get_shape();
                            const size_t token_bytes = inputs[port].get_byte_size() / (B * T);
                            shape[1] = prefix;
                            ov::Tensor sliced(type, shape);
                            for (size_t batch = 0; batch < B; ++batch)
                                std::memcpy(static_cast<char*>(sliced.data()) + batch * prefix * token_bytes,
                                            static_cast<const char*>(inputs[port].data()) + batch * T * token_bytes,
                                            prefix * token_bytes);
                            expected_req.set_input_tensor(port, sliced);
                        }
                        expected_req.infer();
                    }
                    auto states = expected_req.get_output_tensor(1);
                    for (size_t batch = 0; batch < B; ++batch) {
                        const int r = slot * B + batch;
                        if (r < source)
                            continue;
                        const int dst = begin + ((r - source) / sequences) * stride + (r - source) % sequences;
                        for (int h = 0; h < H; ++h)
                            for (int k = 0; k < K; ++k)
                                for (int v = 0; v < V; ++v)
                                    set_value(expected[1], dst * F + (h * V + v) * K + k,
                                              get_value(states, ((batch * H + h) * K + k) * V + v));
                    }
                }
            }
            for (size_t port = 0; port < inputs.size(); ++port) {
                if (port != cache_port)
                    req.set_input_tensor(port, inputs[port]);
            }
            auto remote = host_usm ? context.create_usm_host_tensor(cache_type, inputs[cache_port].get_shape())
                                   : context.create_usm_device_tensor(cache_type, inputs[cache_port].get_shape());
            inputs[cache_port].copy_to(remote);
            req.set_input_tensor(cache_port, remote);
            auto other = context.create_usm_device_tensor(cache_type, remote.get_shape());
            if (unrelated_overlap) {
                inputs[cache_port].copy_to(other);
                req.set_input_tensor(inputs.size(), other);
            }
            // Alternate distinct and identical USM allocations on one request.
            req.set_output_tensor(1, unrelated_overlap ? other : step % 3 == 2 ? context.create_usm_device_tensor(cache_type, remote.get_shape()) : remote);
            req.infer();
            for (size_t port = 0; port < 2; ++port) {
                auto out = req.get_output_tensor(port), reference = expected[port];
                ov::Tensor actual(out.get_element_type(), out.get_shape());
                out.copy_to(actual);
                ASSERT_EQ(actual.get_shape(), reference.get_shape());
                for (size_t i = 0; i < actual.get_size(); ++i) {
                    const auto get = [i](const ov::Tensor& t) {
                        return t.get_element_type() == ov::element::f16 ? float(t.data<ov::float16>()[i]) : t.data<float>()[i];
                    };
                    ASSERT_NEAR(get(actual), get(reference), conv ? 1e-6f : type == ov::element::f16 ? 0.002f : 1e-6f) << "conv=" << conv << " step=" << step << " port=" << port << " element=" << i;
                }
            }
            if (other_reader) {
                auto reader = req.get_output_tensor(2);
                ov::Tensor host(ov::element::f32, reader.get_shape());
                reader.copy_to(host);
                for (size_t i = 0; i < host.get_size(); ++i)
                    ASSERT_EQ(host.data<float>()[i], inputs[cache_port].data<float>()[i]);
            }
            if (unrelated_overlap) {
                auto branch = req.get_output_tensor(2);
                ov::Tensor host(ov::element::f32, branch.get_shape());
                branch.copy_to(host);
                for (size_t i = 0; i < host.get_size(); ++i) {
                    float x = inputs[cache_port].data<float>()[i];
                    ASSERT_EQ(host.data<float>()[i], x * x);
                }
            }
            if (step % 3 == 2 || unrelated_overlap) {
                ov::Tensor host(cache_type, remote.get_shape());
                remote.copy_to(host);
                for (size_t i = 0; i < host.get_size(); ++i)
                    ASSERT_EQ(host.get_element_type() == ov::element::f16 ? float(host.data<ov::float16>()[i]) : host.data<float>()[i],
                              cache_type == ov::element::f16 ? float(inputs[cache_port].data<ov::float16>()[i]) : inputs[cache_port].data<float>()[i]);
            }
        }
    }
}
}  // namespace

TEST(RecurrentCacheUpdateNative, ConvolutionCyclesWindowsAndExport) {
    for (auto type : {ov::element::f16, ov::element::f32})
        for (auto indices : {ov::element::i32, ov::element::i64})
            run_update_case(true, type, indices, 0, 0, 1);
}
TEST(RecurrentCacheUpdateNative, GdnCyclesSnapshotsAndExport) {
    for (auto type : {ov::element::f16, ov::element::f32})
        for (auto indices : {ov::element::i32, ov::element::i64})
            for (int count : {1, 3})
                for (const auto& dims : {std::pair<int, int>{128, 8}, std::pair<int, int>{8, 7}})
                    run_update_case(false, type, indices, dims.first, dims.second, count);
}
TEST(RecurrentCacheUpdateNative, SharedCacheReaderForbidsMutation) {
    run_update_case(true, ov::element::f16, ov::element::i32, 0, 0, 1, true);
    run_update_case(false, ov::element::f16, ov::element::i32, 128, 8, 3, true);
}

TEST(RecurrentCacheUpdateNative, UnrelatedInputOverlapForbidsMutation) {
    run_update_case(true, ov::element::f16, ov::element::i32, 0, 0, 1, false, true);
    run_update_case(false, ov::element::f16, ov::element::i32, 128, 8, 3, false, true);
}

TEST(RecurrentCacheUpdateNative, F16CachePrecision) {
    for (auto compute : {ov::element::f16, ov::element::f32}) {
        run_update_case(true, compute, ov::element::i32, 0, 0, 1, false, false, ov::element::f16);
        run_update_case(false, compute, ov::element::i32, 128, 8, 3, false, false, ov::element::f16);
    }
}

TEST(RecurrentCacheUpdateNative, ExclusiveReshapeCacheAlias) {
    run_update_case(true, ov::element::f16, ov::element::i32, 0, 0, 1, false, false, ov::element::f32, true);
    run_update_case(false, ov::element::f16, ov::element::i32, 128, 8, 3, false, false, ov::element::f32, true);
    // An independent root reader must still veto mutation through a view.
    run_update_case(true, ov::element::f16, ov::element::i32, 0, 0, 1, true, false, ov::element::f32, true);
    run_update_case(false, ov::element::f16, ov::element::i32, 128, 8, 3, true, false, ov::element::f32, true);
}

// Match llama.cpp's default FP32 cache reshapes with FP16 compute. Full-cache
// precision reorders defeat direct USM binding even if numerical tests still pass.
TEST(RecurrentCacheUpdateNative, Fp16InferencePreservesReshapedCachePrecision) {
    using namespace ov;
    auto cache = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 1, -1, 21});
    auto rows = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1, 1, 1, -1});
    auto data = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, -1, 7, -1});
    auto shape = op::v0::Constant::create(element::i64, Shape{4}, {1, 1, -1, 21});
    auto view = std::make_shared<op::v1::Reshape>(cache, shape, false);
    auto zero = op::v0::Constant::create(element::i32, Shape{1, 1, 1, 1}, {0});
    auto offsets = op::v0::Constant::create(element::i32, Shape{1, 1, 1, 2}, {0, 0});
    auto windows = op::v0::Constant::create(element::i32, Shape{1, 1, 1, 2}, {0, 0});
    auto weights = op::v0::Constant::create(element::f16, Shape{1, 1, 7, 4}, {0.1f});
    auto update = std::make_shared<op::internal::RecurrentConvCacheUpdate>(
        OutputVector{view, rows, rows, zero, zero, offsets, data, windows, weights}, 3);
    auto model = std::make_shared<Model>(OutputVector{update->output(0), update->output(1)},
                                       ParameterVector{cache, rows, data});
    Core core;
    auto compiled = core.compile_model(model, "GPU", hint::inference_precision(element::f16));
    auto runtime = compiled.get_runtime_model();
    bool found = false;
    for (const auto& node : runtime->get_ordered_ops()) {
        const auto& info = node->get_rt_info();
        auto type = info.find("layerType");
        if (type == info.end() || type->second.as<std::string>() != "recurrent_cache_update")
            continue;
        ASSERT_EQ(node->get_input_element_type(0), element::f32);
        ASSERT_EQ(compiled.output(1).get_element_type(), element::f32);
        found = true;
    }
    ASSERT_TRUE(found);
}

// Standalone snapshots use a temporary bank, then ordinary scatter writeback.
// Its precision must follow the surrounding computation under FP16 inference.
TEST(RecurrentCacheUpdateNative, Fp16InferenceWithTemporarySnapshotBank) {
    using namespace ov;
    ParameterVector params;
    auto param = [&](PartialShape shape) {
        auto p = std::make_shared<op::v0::Parameter>(element::f32, shape);
        params.push_back(p);
        return p;
    };
    auto q = param({1, 2, 2, 8}), k = param({1, 2, 2, 8}), v = param({1, 2, 2, 8});
    auto cache = param({1, 1, 3, 128});
    auto gate = param({1, 2, 2}), beta = param({1, 2, 2});
    auto index = [](Shape shape, std::vector<int32_t> values) {
        return op::v0::Constant::create(element::i32, shape, values);
    };
    auto axis = index({1}, {2}), begin = index({1}, {0}), step = index({1}, {1});
    auto initial = std::make_shared<op::v8::Slice>(cache, begin, step, step, axis);
    auto bank = std::make_shared<op::v0::Concat>(
        OutputVector{initial, op::v0::Constant::create(element::f32, Shape{1, 1, 2, 128}, {0})}, 2);
    auto zero = index({1, 1, 1, 1}, {0});
    auto update = std::make_shared<op::internal::RecurrentGatedDeltaNetUpdate>(
        OutputVector{q, k, v, bank, gate, beta, index({1, 1, 1, 1}, {0}), index({1, 1, 1, 0}, {}),
                     zero, zero, index({1, 1, 1, 2}, {0, 0}), index({1, 1, 1, 4}, {0, 1, 1, 0})},
        false, 1e-6f, 1e-6f, 3);
    auto states = std::make_shared<op::v8::Slice>(update->output(1), begin, index({1}, {2}), step, axis);
    auto writeback = std::make_shared<op::v3::ScatterUpdate>(cache, index({2}, {0, 1}), states, axis);
    Core core;
    auto model = std::make_shared<Model>(OutputVector{update->output(0), writeback}, params);
    auto compiled = core.compile_model(model, "GPU", hint::inference_precision(element::f16));
    bool found = false;
    for (const auto& node : compiled.get_runtime_model()->get_ordered_ops()) {
        const auto& info = node->get_rt_info();
        auto type = info.find("layerType");
        if (type == info.end() || type->second.as<std::string>() != "recurrent_cache_update")
            continue;
        ASSERT_EQ(node->get_input_element_type(3), element::f16);
        found = true;
    }
    ASSERT_TRUE(found);
    ASSERT_EQ(compiled.output(1).get_element_type(), element::f32);
}

// Integrated GPUs can expose caller cache storage as host USM. Apply the same
// exclusive-alias contract without staging the full cache through device USM.
TEST(RecurrentCacheUpdateNative, HostUsmExclusiveCacheAlias) {
    run_update_case(true, ov::element::f16, ov::element::i32, 0, 0, 1, false, false, ov::element::f32, true, true);
    run_update_case(false, ov::element::f16, ov::element::i32, 128, 8, 3, false, false, ov::element::f32, true, true);
}
