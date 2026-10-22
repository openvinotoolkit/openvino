// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/op/recurrent_cache_update.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <sstream>
#include <vector>

#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/recurrent_cache.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"
#include "openvino/runtime/properties.hpp"

namespace {
// Keep the reference in a separate graph: sharing its cache Parameter with the
// Update op would correctly forbid mutation and only exercise the copy fallback.
void run_update_case(bool conv,
                     ov::element::Type type,
                     ov::element::Type indices,
                     int K,
                     int V,
                     int snapshots,
                     bool other_reader = false,
                     bool unrelated_overlap = false,
                     ov::element::Type cache_type = ov::element::f32,
                     bool cache_view = false) {
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
    std::shared_ptr<ov::Node> reference, update;
    if (conv) {
        reference = std::make_shared<ov::op::internal::RecurrentConvCache>(args, history);
        update = std::make_shared<ov::op::internal::RecurrentConvCacheUpdate>(args, history);
    } else {
        reference = std::make_shared<ov::op::internal::RecurrentGatedDeltaNet>(args, true, 1e-6f, 1e-6f, snapshots);
        update = std::make_shared<ov::op::internal::RecurrentGatedDeltaNetUpdate>(args, true, 1e-6f, 1e-6f, snapshots);
    }
    ov::Core core;
    const auto hint = ov::hint::inference_precision(ov::element::dynamic);
    auto ref = core.compile_model(std::make_shared<ov::Model>(ov::OutputVector{reference->output(0), reference->output(1)}, params), "GPU", hint);
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
        auto req = model.create_infer_request(), expected_req = ref.create_infer_request();
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
            for (size_t port = 0; port < inputs.size(); ++port) {
                expected_req.set_input_tensor(port, inputs[port]);
                if (port != cache_port)
                    req.set_input_tensor(port, inputs[port]);
            }
            expected_req.infer();
            auto remote = context.create_usm_device_tensor(cache_type, inputs[cache_port].get_shape());
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
                auto out = req.get_output_tensor(port), expected = expected_req.get_output_tensor(port);
                ov::Tensor actual(out.get_element_type(), out.get_shape());
                out.copy_to(actual);
                ASSERT_EQ(actual.get_shape(), expected.get_shape());
                for (size_t i = 0; i < actual.get_size(); ++i) {
                    const auto get = [i](const ov::Tensor& t) {
                        return t.get_element_type() == ov::element::f16 ? float(t.data<ov::float16>()[i]) : t.data<float>()[i];
                    };
                    ASSERT_EQ(get(actual), get(expected)) << "conv=" << conv << " step=" << step << " port=" << port << " element=" << i;
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
