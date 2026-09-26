// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/eltwise.hpp"
#include "internal_properties.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"
#include "nodes/kernels/riscv64/cpu_isa_traits.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/power.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Eltwise {
namespace {

/*
 * The motivation of this test is to validate different input and output precisions of Eltwise Op.
 * If IO data type is not supported by jit emitter, they should be converted
 * to supported types on operation inputs and outputs in JIT kernel
*/

static const std::vector<ov::test::utils::EltwiseTypes> ops() {
    // JIT us supported only when `gv` is available
    if (ov::intel_cpu::riscv64::mayiuse(ov::intel_cpu::riscv64::gv)) {
        return { utils::EltwiseTypes::ADD };
    }
    return {};
}

const std::vector<ov::AnyMap>& config_infer_prc_f32() {
    static const std::vector<ov::AnyMap> additionalConfig = {
        {{ov::hint::inference_precision.name(), ov::element::f32}},
    };
    return additionalConfig;
}

const std::vector<std::vector<ov::Shape>>& inputShapes() {
    static const std::vector<std::vector<ov::Shape>> inputShapes = {
        {{2, 4, 4, 1}},
        {{2, 4, 4, 128}},
        {{2, 4, 4, 131}},
        {{2, 17, 5, 4}},
        {{2, 19, 5, 4}, {1, 19, 1, 1}},
        {{2, 19, 5, 1}, {1, 19, 1, 4}},
    };
    return inputShapes;
}

const auto params_4D_jit = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes())),
                ::testing::ValuesIn(ops()),
                ::testing::ValuesIn(secondaryInputTypes()),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn({ ElementType::i8, ElementType::u8, ElementType::f16, ElementType::i32, ElementType::f32 }),
                ::testing::Values(ov::element::dynamic),
                ::testing::Values(ov::element::dynamic),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(config_infer_prc_f32())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D())),
        ::testing::Values(emptyFusingSpec),
        ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_jit, EltwiseLayerCPUTest, params_4D_jit, EltwiseLayerCPUTest::getTestCaseName);

class PowerStaticRvvLmulTest : virtual public SubgraphBaseTest, public CpuTestWithFusing {
protected:
    void SetUp() override {
        if (!ov::intel_cpu::riscv64::mayiuse(ov::intel_cpu::riscv64::gv)) {
            GTEST_SKIP();
        }

        targetDevice = ov::test::utils::DEVICE_CPU;
        const auto shape = ov::PartialShape{1, 1, 1, 35};
        init_input_shapes({{shape, {shape.to_shape()}}});

        auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
        auto exponent = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, 1.5F);
        auto power = std::make_shared<ov::op::v1::Power>(parameter, exponent);
        ov::ParameterVector parameters{parameter};
        function = create_ov_model(ov::element::f32, parameters, power, "PowerStaticRvvLmul");
        configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::IGNORE_CALLBACK));
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& parameter = function->get_parameters().front();
        const ov::test::utils::InputGenerateData input_data(1, 2, 1);
        inputs.insert({parameter, ov::test::utils::create_and_fill_tensor(ov::element::f32,
                                                                          targetInputStaticShapes.front(),
                                                                          input_data)});
    }
};

TEST_F(PowerStaticRvvLmulTest, CompareWithRefs) {
    run();
}

}  // namespace
}  // namespace Eltwise
}  // namespace test
}  // namespace ov
