// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <tuple>

#include "openvino/core/any.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"
#include "utils/precision_support.h"

namespace ov::test {
namespace {

using SelectiveSSMJitParams = std::tuple<bool, ov::element::Type>;

std::shared_ptr<ov::Model> make_selective_ssm_model(const ov::element::Type& precision,
                                                    bool paged,
                                                    size_t state_size = 16,
                                                    size_t tokens = 1) {
    const auto A = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{4});
    const auto state =
        std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{paged ? size_t{2} : size_t{1}, 4, 5, state_size});

    ov::ParameterVector parameters;
    std::shared_ptr<ov::Node> operation;
    if (paged) {
        const auto dt = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{tokens, 4});
        const auto B = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{tokens, 2, state_size});
        const auto x = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{tokens, 4, 5});
        const auto C = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{tokens, 2, state_size});
        const auto subsequences = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2});
        const auto blocks = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2});
        const auto block_begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2});
        const auto processed = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
        const auto intervals = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
        parameters = {A, dt, B, x, C, state, subsequences, blocks, block_begins, processed, intervals};
        operation = std::make_shared<ov::op::internal::PagedSelectiveSSM>(parameters[0],
                                                                          parameters[1],
                                                                          parameters[2],
                                                                          parameters[3],
                                                                          parameters[4],
                                                                          parameters[5],
                                                                          parameters[6],
                                                                          parameters[7],
                                                                          parameters[8],
                                                                          parameters[9],
                                                                          parameters[10]);
    } else {
        const auto dt = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{1, tokens, 4});
        const auto B = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{1, tokens, 2, state_size});
        const auto x = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{1, tokens, 4, 5});
        const auto C = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{1, tokens, 2, state_size});
        parameters = {A, dt, B, x, C, state};
        operation = std::make_shared<ov::op::internal::SelectiveSSM>(parameters[0],
                                                                     parameters[1],
                                                                     parameters[2],
                                                                     parameters[3],
                                                                     parameters[4],
                                                                     parameters[5]);
    }

    operation->set_friendly_name("ssm");
    return std::make_shared<ov::Model>(operation->outputs(), parameters);
}

class SelectiveSSMJitIntegrationTest : public testing::TestWithParam<SelectiveSSMJitParams> {};

template <typename T>
void fill_tensor(ov::Tensor& tensor, float value) {
    std::fill_n(tensor.data<T>(), tensor.get_size(), static_cast<T>(value));
}

void fill_data_tensor(ov::Tensor& tensor, float value) {
    if (tensor.get_element_type() == ov::element::f32) {
        fill_tensor<float>(tensor, value);
    } else if (tensor.get_element_type() == ov::element::f16) {
        fill_tensor<ov::float16>(tensor, value);
    } else {
        fill_tensor<ov::bfloat16>(tensor, value);
    }
}

float tensor_value(const ov::Tensor& tensor, size_t index) {
    if (tensor.get_element_type() == ov::element::f32) {
        return tensor.data<const float>()[index];
    }
    if (tensor.get_element_type() == ov::element::f16) {
        return static_cast<float>(tensor.data<const ov::float16>()[index]);
    }
    return static_cast<float>(tensor.data<const ov::bfloat16>()[index]);
}

void fill_inputs(ov::InferRequest& request, bool paged, const std::array<float, 6>& values) {
    for (size_t i = 0; i < values.size(); ++i) {
        auto tensor = request.get_input_tensor(i);
        fill_data_tensor(tensor, values[i]);
    }
    if (paged) {
        const auto tokens = static_cast<int32_t>(request.get_input_tensor(3).get_shape()[0]);
        const int32_t metadata[][2] = {{0, tokens}, {0, 1}, {0, 2}, {0, 0}, {tokens, 0}};
        for (size_t i = 0; i < 5; ++i) {
            auto tensor = request.get_input_tensor(6 + i);
            std::copy_n(metadata[i], tensor.get_size(), tensor.data<int32_t>());
        }
    }
}

void expect_selected_executor(const ov::CompiledModel& compiled_model,
                              bool paged,
                              const ov::element::Type& precision,
                              bool jit_state_size = true) {
    const auto runtime_model = compiled_model.get_runtime_model();

    const auto expected_layer = paged ? std::string{"PagedSelectiveSSM"} : std::string{"SelectiveSSM"};
    // Match the executor's effective ISA, including oneDNN's runtime ISA limit.
    using namespace dnnl::impl::cpu::x64;
    const bool native_precision = precision == ov::element::f32   ? mayiuse(avx2)
                                  : precision == ov::element::f16 ? mayiuse(avx512_core_fp16) || mayiuse(avx2_vnni_2)
                                                                  : mayiuse(avx512_core_bf16) || mayiuse(avx2_vnni_2);
    const bool native = native_precision && jit_state_size;
    const auto expected_implementation =
        std::string{mayiuse(avx512_core) ? "jit_avx512_" : "jit_avx2_"} + precision.get_type_name();
    size_t matching_nodes = 0;
    for (const auto& node : runtime_model->get_ops()) {
        const auto& rt_info = node->get_rt_info();
        const auto layer = rt_info.find(ov::exec_model_info::LAYER_TYPE);
        if (layer == rt_info.end() || layer->second.as<std::string>() != expected_layer) {
            continue;
        }

        ++matching_nodes;
        const auto implementation = rt_info.find(ov::exec_model_info::IMPL_TYPE);
        ASSERT_NE(implementation, rt_info.end());
        const auto actual_implementation = implementation->second.as<std::string>();
        if (native) {
            EXPECT_EQ(actual_implementation, expected_implementation);
        } else {
            EXPECT_EQ(actual_implementation.find("ref"), 0U) << actual_implementation;
        }
        EXPECT_EQ(node->get_output_element_type(0), precision);
    }
    EXPECT_EQ(matching_nodes, 1U);
}

TEST_P(SelectiveSSMJitIntegrationTest, InfersWithSelectedExecutorAndPreservedDataPrecision) {
    const auto& [paged, precision] = GetParam();
    if (!ov::intel_cpu::hasHardwareSupport(precision)) {
        GTEST_SKIP() << "CPU precision policy does not preserve " << precision << " on this system";
    }

    ov::Core core;
    const ov::AnyMap properties{{ov::hint::inference_precision.name(), precision}};
    auto compiled_model = core.compile_model(make_selective_ssm_model(precision, paged), "CPU", properties);
    expect_selected_executor(compiled_model, paged, precision);

    auto request = compiled_model.create_infer_request();
    // Exactly representable values give an independent oracle for both executors and all data precisions:
    // decay=exp(0)=1; state=0.5 + (0.5 * 2) * 0.25=0.75; output=16 * 0.75 * 0.125=1.5.
    fill_inputs(request, paged, {0.F, 0.5F, 0.25F, 2.F, 0.125F, 0.5F});
    request.infer();
    const auto output = request.get_output_tensor(0);
    EXPECT_EQ(output.get_element_type(), precision);
    for (size_t i = 0; i < output.get_size(); ++i) {
        EXPECT_FLOAT_EQ(tensor_value(output, i), 1.5F) << "output index " << i;
    }
    if (paged) {
        const auto cache = request.get_input_tensor(5);
        const auto block_size = cache.get_size() / 2;
        for (size_t i = 0; i < block_size; ++i) {
            EXPECT_FLOAT_EQ(tensor_value(cache, i), 0.5F) << "read block index " << i;
            EXPECT_FLOAT_EQ(tensor_value(cache, block_size + i), 0.75F) << "snapshot index " << i;
        }
    } else {
        const auto state = request.get_output_tensor(1);
        EXPECT_EQ(state.get_element_type(), precision);
        for (size_t i = 0; i < state.get_size(); ++i) {
            EXPECT_FLOAT_EQ(tensor_value(state, i), 0.75F) << "state index " << i;
        }
    }
}

TEST_P(SelectiveSSMJitIntegrationTest, StateSizeLimitSelectsJitOrReferenceWithCorrectRecurrence) {
    const auto& [paged, precision] = GetParam();
    if (!ov::intel_cpu::hasHardwareSupport(precision)) {
        GTEST_SKIP() << "CPU precision policy does not preserve " << precision << " on this system";
    }

    ov::Core core;
    const auto round_output = [&](float value) {
        if (precision == ov::element::f16) {
            return static_cast<float>(ov::float16(value));
        }
        if (precision == ov::element::bf16) {
            return static_cast<float>(ov::bfloat16(value));
        }
        return value;
    };
    // Check the advertised maximum and the reference fallback beyond it through both decode and prefill.
    for (const size_t state_size : {4096U, 4097U}) {
        for (const size_t tokens : {1U, 3U}) {
            SCOPED_TRACE(testing::Message() << "state_size=" << state_size << " tokens=" << tokens);
            auto compiled = core.compile_model(make_selective_ssm_model(precision, paged, state_size, tokens),
                                               "CPU",
                                               ov::hint::inference_precision(precision));
            expect_selected_executor(compiled, paged, precision, state_size == 4096);
            auto request = compiled.create_infer_request();
            fill_inputs(request, paged, {0.F, 0.5F, 0.25F, 2.F, 0.125F, 0.5F});
            request.infer();
            const auto output = request.get_output_tensor(0);
            ASSERT_EQ(output.get_size(), tokens * 20);
            for (size_t i = 0; i < output.get_size(); ++i) {
                const auto state_value = 0.75F + 0.25F * static_cast<float>(i / 20);
                EXPECT_FLOAT_EQ(tensor_value(output, i), round_output(state_value * 0.125F * state_size))
                    << "output index " << i;
            }
            const auto state = paged ? request.get_input_tensor(5) : request.get_output_tensor(1);
            const auto block_size = paged ? state.get_size() / 2 : state.get_size();
            const auto final_value = 0.5F + 0.25F * static_cast<float>(tokens);
            for (size_t i = 0; i < block_size; ++i) {
                if (paged) {
                    EXPECT_FLOAT_EQ(tensor_value(state, i), 0.5F) << "read block index " << i;
                }
                EXPECT_FLOAT_EQ(tensor_value(state, paged ? block_size + i : i), final_value)
                    << "final state index " << i;
            }
        }
    }
}

TEST_P(SelectiveSSMJitIntegrationTest, SerialRecurrenceWorksOnFreshThread) {
    const auto& [paged, precision] = GetParam();
    if (!ov::intel_cpu::hasHardwareSupport(precision)) {
        GTEST_SKIP() << "CPU precision policy does not preserve " << precision << " on this system";
    }

    ov::Core core;
    // With no streams or parallel workers, inference runs on the calling thread outside a TBB arena.
    auto compiled = core.compile_model(make_selective_ssm_model(precision, paged, 16, 3),
                                       "CPU",
                                       ov::hint::inference_precision(precision),
                                       ov::num_streams(0),
                                       ov::inference_num_threads(1));
    std::thread worker([&] {
        EXPECT_NO_THROW({
            for (const bool alias_state : {false, true}) {
                if (paged && alias_state) {
                    continue;
                }
                SCOPED_TRACE(testing::Message() << "alias_state=" << alias_state);
                auto request = compiled.create_infer_request();
                fill_inputs(request, paged, {0.F, 0.5F, 0.25F, 2.F, 0.125F, 0.5F});
                if (alias_state) {
                    request.set_output_tensor(1, request.get_input_tensor(5));
                }
                request.infer();
                const auto output = request.get_output_tensor(0);
                // Each token adds 0.25 to state; all 20 head/row positions have the same exact output.
                for (size_t i = 0; i < output.get_size(); ++i) {
                    EXPECT_FLOAT_EQ(tensor_value(output, i), 1.5F + 0.5F * static_cast<float>(i / 20));
                }
                const auto state = paged ? request.get_input_tensor(5) : request.get_output_tensor(1);
                const auto state_size = paged ? state.get_size() / 2 : state.get_size();
                for (size_t i = 0; i < state_size; ++i) {
                    if (paged) {
                        EXPECT_FLOAT_EQ(tensor_value(state, i), 0.5F) << "read block index " << i;
                    }
                    EXPECT_FLOAT_EQ(tensor_value(state, paged ? state_size + i : i), 1.25F)
                        << "final state index " << i;
                }
            }
        });
    });
    worker.join();
}

TEST(PagedSelectiveSSMJitIntegrationTest, HandlesInt32CacheIntervalExtremes) {
    ov::Core core;
    for (const bool store_snapshot : {false, true}) {
        SCOPED_TRACE(testing::Message() << "store_snapshot=" << store_snapshot);
        const size_t tokens = store_snapshot ? 1 : 2;
        auto compiled = core.compile_model(make_selective_ssm_model(ov::element::f32, true, 5, tokens),
                                           "CPU",
                                           ov::hint::inference_precision(ov::element::f32));
        auto request = compiled.create_infer_request();
        fill_inputs(request, true, {0.F, 0.5F, 0.25F, 2.F, 0.125F, 0.5F});
        request.get_input_tensor(9).data<int32_t>()[0] = std::numeric_limits<int32_t>::max();
        request.get_input_tensor(10).data<int32_t>()[0] =
            store_snapshot ? std::numeric_limits<int32_t>::max() : std::numeric_limits<int32_t>::min();
        request.infer();
        const auto output = request.get_output_tensor(0);
        for (size_t i = 0; i < output.get_size(); ++i) {
            EXPECT_FLOAT_EQ(tensor_value(output, i), 5.F * 0.125F * (0.75F + 0.25F * static_cast<float>(i / 20)));
        }
        const auto cache = request.get_input_tensor(5);
        const auto block_size = cache.get_size() / 2;
        for (size_t i = 0; i < block_size; ++i) {
            EXPECT_FLOAT_EQ(tensor_value(cache, i), 0.5F) << "read block index " << i;
            EXPECT_FLOAT_EQ(tensor_value(cache, block_size + i), store_snapshot ? 0.75F : 0.5F)
                << "snapshot index " << i;
        }
    }
}

TEST(PagedSelectiveSSMJitIntegrationTest, DisabledCacheAllowsSharedReadBlocks) {
    auto model = make_selective_ssm_model(ov::element::f32, true, 5, 3);
    const auto& params = model->get_parameters();
    params[6]->set_partial_shape(ov::Shape{3});
    params[8]->set_partial_shape(ov::Shape{3});
    params[9]->set_partial_shape(ov::Shape{2});
    params[10]->set_partial_shape(ov::Shape{2});
    model->validate_nodes_and_infer_types();
    ov::Core core;
    auto compiled = core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32));
    auto request = compiled.create_infer_request();
    for (size_t i = 0; i < 6; ++i) {
        auto tensor = request.get_input_tensor(i);
        const std::array<float, 6> values{0.F, 0.5F, 0.25F, 2.F, 0.125F, 0.5F};
        fill_data_tensor(tensor, values[i]);
    }
    const int32_t metadata[][3] = {{0, 1, 3}, {0, 0}, {0, 1, 2}, {7, 9}, {0, -3}};
    for (size_t i = 0; i < 5; ++i) {
        auto tensor = request.get_input_tensor(6 + i);
        std::copy_n(metadata[i], tensor.get_size(), tensor.data<int32_t>());
    }
    request.infer();
    const auto output = request.get_output_tensor(0);
    for (size_t i = 0; i < output.get_size(); ++i) {
        const auto state_value = i / 20 == 2 ? 1.F : 0.75F;
        EXPECT_FLOAT_EQ(tensor_value(output, i), 5.F * 0.125F * state_value) << "output index " << i;
    }
    const auto cache = request.get_input_tensor(5);
    for (size_t i = 0; i < cache.get_size(); ++i) {
        EXPECT_FLOAT_EQ(tensor_value(cache, i), 0.5F) << "unchanged cache index " << i;
    }
}

TEST(PagedSelectiveSSMJitIntegrationTest, EmptySubsequenceListPreservesCache) {
    auto model = make_selective_ssm_model(ov::element::f32, true, 5, 0);
    const auto& params = model->get_parameters();
    params[6]->set_partial_shape(ov::Shape{1});
    params[7]->set_partial_shape(ov::Shape{0});
    params[8]->set_partial_shape(ov::Shape{1});
    params[9]->set_partial_shape(ov::Shape{0});
    params[10]->set_partial_shape(ov::Shape{0});
    model->validate_nodes_and_infer_types();
    ov::Core core;
    auto compiled = core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::f32));
    auto request = compiled.create_infer_request();
    fill_inputs(request, true, {0.F, 0.5F, 0.25F, 2.F, 0.125F, 0.5F});
    request.infer();
    EXPECT_EQ(request.get_output_tensor(0).get_size(), 0U);
    const auto cache = request.get_input_tensor(5);
    for (size_t i = 0; i < cache.get_size(); ++i) {
        EXPECT_FLOAT_EQ(tensor_value(cache, i), 0.5F) << "unchanged cache index " << i;
    }
}

TEST(SelectiveSSMJitTailTest, InactiveLanesDoNotTurnInfiniteOutputIntoNaN) {
    using namespace dnnl::impl::cpu::x64;
    if (!mayiuse(avx2)) {
        GTEST_SKIP() << "Requires the SelectiveSSM JIT executor";
    }
    ov::Core core;
    const std::array<std::array<float, 6>, 2> cases{{
        {100.F, 1.F, 0.F, 0.F, 1.F, 1.F},  // exp(A * delta) overflows; inactive state lanes contain 0 * Inf.
        {0.F, 1.F, 1.F, std::numeric_limits<float>::infinity(), 1.F, 1.F},  // Inactive B lanes contain 0 * Inf.
    }};
    for (const bool paged : {false, true}) {
        for (const size_t state_size : {1U, 3U, 9U, 17U}) {
            auto compiled = core.compile_model(make_selective_ssm_model(ov::element::f32, paged, state_size),
                                               "CPU",
                                               ov::hint::inference_precision(ov::element::f32));
            for (const auto& values : cases) {
                SCOPED_TRACE(testing::Message()
                             << "paged=" << paged << " state_size=" << state_size << " A=" << values[0]);
                auto request = compiled.create_infer_request();
                fill_inputs(request, paged, values);
                request.infer();
                const auto output = request.get_output_tensor(0);
                for (size_t i = 0; i < output.get_size(); ++i) {
                    const auto value = tensor_value(output, i);
                    EXPECT_TRUE(std::isinf(value) && value > 0.F) << "output index " << i << ": " << value;
                }
            }
        }
    }
}

std::string selective_ssm_jit_test_name(const testing::TestParamInfo<SelectiveSSMJitParams>& info) {
    const auto& [paged, precision] = info.param;
    return std::string{paged ? "Paged" : "Selective"} + "_" + precision.get_type_name();
}

INSTANTIATE_TEST_SUITE_P(smoke_SelectiveSSMJit,
                         SelectiveSSMJitIntegrationTest,
                         testing::Combine(testing::Bool(),
                                          testing::Values(ov::element::f32, ov::element::f16, ov::element::bf16)),
                         selective_ssm_jit_test_name);

}  // namespace
}  // namespace ov::test
