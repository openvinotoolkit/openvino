// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <string>
#include <tuple>

#include "openvino/core/any.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/selective_ssm.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/system_conf.hpp"
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
        const auto subsequences = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{2});
        const auto blocks = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{2});
        const auto block_begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{2});
        const auto processed = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
        const auto intervals = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
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

TEST_P(SelectiveSSMJitIntegrationTest, SelectsJitWithoutWideningDataPrecision) {
    if (!ov::with_cpu_x86_avx2()) {
        GTEST_SKIP() << "SelectiveSSM JIT requires AVX2 or newer";
    }

    const auto& [paged, precision] = GetParam();
    if (!ov::intel_cpu::hasHardwareSupport(precision)) {
        GTEST_SKIP() << "CPU precision policy does not preserve " << precision << " on this system";
    }

    ov::Core core;
    const ov::AnyMap properties{{ov::hint::inference_precision.name(), precision}};
    const auto compiled_model = core.compile_model(make_selective_ssm_model(precision, paged), "CPU", properties);
    const auto runtime_model = compiled_model.get_runtime_model();

    const auto expected_layer = paged ? std::string{"PagedSelectiveSSM"} : std::string{"SelectiveSSM"};
    const auto expected_implementation =
        std::string{ov::with_cpu_x86_avx512f() ? "jit_avx512_" : "jit_avx2_"} + precision.get_type_name();
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
        EXPECT_EQ(implementation->second.as<std::string>(), expected_implementation);
        EXPECT_EQ(node->get_output_element_type(0), precision);
    }
    EXPECT_EQ(matching_nodes, 1U);
}

std::string selective_ssm_jit_test_name(const testing::TestParamInfo<SelectiveSSMJitParams>& info) {
    const auto& [paged, precision] = info.param;
    return std::string{paged ? "Paged" : "Selective"} + "_" + precision.get_type_name();
}

INSTANTIATE_TEST_SUITE_P(smoke_SelectiveSSMJit,
                         SelectiveSSMJitIntegrationTest,
                         testing::Combine(testing::Bool(), testing::Values(ov::element::f16, ov::element::bf16)),
                         selective_ssm_jit_test_name);

}  // namespace
}  // namespace ov::test
