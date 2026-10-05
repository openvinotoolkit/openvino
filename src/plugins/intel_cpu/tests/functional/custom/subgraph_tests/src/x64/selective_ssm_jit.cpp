// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
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

std::shared_ptr<ov::Model> make_selective_ssm_model(const ov::element::Type& precision, bool paged) {
    const auto A = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{4});
    const auto state =
        std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{paged ? size_t{2} : size_t{1}, 4, 5, 16});

    ov::ParameterVector parameters;
    std::shared_ptr<ov::Node> operation;
    if (paged) {
        const auto dt = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{1, 4});
        const auto B = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{1, 2, 16});
        const auto x = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{1, 4, 5});
        const auto C = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{1, 2, 16});
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
        const auto dt = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{1, 1, 4});
        const auto B = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{1, 1, 2, 16});
        const auto x = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{1, 1, 4, 5});
        const auto C = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{1, 1, 2, 16});
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

TEST_P(SelectiveSSMJitIntegrationTest, InfersWithSelectedExecutorAndPreservedDataPrecision) {
    const auto& [paged, precision] = GetParam();
    if (!ov::intel_cpu::hasHardwareSupport(precision)) {
        GTEST_SKIP() << "CPU precision policy does not preserve " << precision << " on this system";
    }

    ov::Core core;
    const ov::AnyMap properties{{ov::hint::inference_precision.name(), precision}};
    auto compiled_model = core.compile_model(make_selective_ssm_model(precision, paged), "CPU", properties);
    const auto runtime_model = compiled_model.get_runtime_model();

    const auto expected_layer = paged ? std::string{"PagedSelectiveSSM"} : std::string{"SelectiveSSM"};
    // Match the executor's effective ISA, including oneDNN's runtime ISA limit.
    using namespace dnnl::impl::cpu::x64;
    const bool native = precision == ov::element::f32   ? mayiuse(avx2)
                        : precision == ov::element::f16 ? mayiuse(avx512_core_fp16) || mayiuse(avx2_vnni_2)
                                                        : mayiuse(avx512_core_bf16) || mayiuse(avx2_vnni_2);
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

    auto request = compiled_model.create_infer_request();
    // Exactly representable values give an independent oracle for both executors and all data precisions:
    // decay=exp(0)=1; state=0.5 + (0.5 * 2) * 0.25=0.75; output=16 * 0.75 * 0.125=1.5.
    constexpr float input_values[] = {0.F, 0.5F, 0.25F, 2.F, 0.125F, 0.5F};
    for (size_t i = 0; i < 6; ++i) {
        auto tensor = request.get_input_tensor(i);
        fill_data_tensor(tensor, input_values[i]);
    }
    if (paged) {
        constexpr int32_t metadata[][2] = {{0, 1}, {0, 1}, {0, 2}, {0, 0}, {1, 0}};
        for (size_t i = 0; i < 5; ++i) {
            auto tensor = request.get_input_tensor(6 + i);
            std::copy_n(metadata[i], tensor.get_size(), tensor.data<int32_t>());
        }
    }
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
