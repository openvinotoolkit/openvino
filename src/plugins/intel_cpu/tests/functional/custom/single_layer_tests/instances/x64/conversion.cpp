// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/conversion.hpp"
#include "openvino/core/type/float_util.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Conversion {
namespace {

std::vector<CPUSpecificParams> memForm4D_dynamic = {
    CPUSpecificParams({nChw8c}, {nChw8c}, {}, "ref"),
    CPUSpecificParams({nChw16c}, {nChw16c}, {}, "ref")
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_blocked_Dynamic, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_dynamic()),
                                ::testing::ValuesIn(precisions()),
                                ::testing::ValuesIn(precisions()),
                                ::testing::Values(ov::test::SpecialValue::none),
                                ::testing::ValuesIn(memForm4D_dynamic)),
                        ConvertCPULayerTest::getTestCaseName);

std::vector<InputShape> inShapes_4D_blocked = {
    {{1, 16, 5, 5}, {{1, 16, 5, 5}}},
};

std::vector<CPUSpecificParams> memForm4D_static_blocked = {
    CPUSpecificParams({nChw16c}, {nChw16c}, {}, {})
};

const std::vector<ov::element::Type> precisions_floating_point = {
        ov::element::f32,
        ov::element::bf16
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_Blocked, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_blocked),
                                ::testing::ValuesIn(precisions()),
                                ::testing::ValuesIn(precisions()),
                                ::testing::Values(ov::test::SpecialValue::none),
                                ::testing::ValuesIn(filterCPUSpecificParams(memForm4D_static_blocked))),
                        ConvertCPULayerTest::getTestCaseName);

class ConvertToFP8RoundingCPULayerTest : public ConvertCPULayerTest {
protected:
    void SetUp() override {
        ConvertCPULayerTest::SetUp();
        abs_threshold = 0.f;
        rel_threshold = 0.f;
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        // f32 values with a tie in f16 but not in f32 (1.0625 + 2^-20, 1.1875 - 2^-20, 2^-10 + 2^-33, 2^-17 + 2^-40,
        // 464 + 2^-15, 61439.996), and NaNs whose payload does not fit to f16
        static const std::vector<uint32_t> values = {0x3F880008, 0xBF880008, 0x3F97FFF8, 0x3A800001,
                                                     0xBA800001, 0x37000001, 0x43E80001, 0x476FFFFF,
                                                     0x7F800001, 0xFF800001, 0xFF9FFFFF, 0x80000000};
        inputs.clear();
        const auto funcInput = function->inputs().front();
        ov::Tensor tensor(funcInput.get_element_type(), targetInputStaticShapes.front());
        auto* data = tensor.data<float>();
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            data[i] = ov::util::u32_bits_to_f32(values[i % values.size()]);
        }
        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }
};

TEST_P(ConvertToFP8RoundingCPULayerTest, CompareWithRefs) {
    run();
}

// 24 elements: one full vector of 16 f32 elements and a tail
const std::vector<InputShape> inShapes_fp8_rounding = {{{1, 2, 3, 4}, {{1, 2, 3, 4}}}};

INSTANTIATE_TEST_SUITE_P(smoke_ConvertToFP8RoundingCPULayerTest, ConvertToFP8RoundingCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_fp8_rounding),
                                ::testing::Values(ov::element::f32),
                                ::testing::Values(ov::element::f8e4m3, ov::element::f8e5m2),
                                ::testing::Values(ov::test::SpecialValue::none),
                                ::testing::Values(CPUSpecificParams({nchw}, {nchw}, {}, "ref"))),
                        ConvertToFP8RoundingCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_BOOL_Static, ConvertToBooleanCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_static()),
                                ::testing::ValuesIn(precisions_floating_point),
                                ::testing::Values(ov::element::boolean),
                                ::testing::Values(ov::test::SpecialValue::none),
                                ::testing::Values(CPUSpecificParams({nchw}, {nchw}, {}, {}))),
                        ConvertToBooleanCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_BOOL_Dynamic, ConvertToBooleanCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_dynamic()),
                                ::testing::ValuesIn(precisions_floating_point),
                                ::testing::Values(ov::element::boolean),
                                ::testing::Values(ov::test::SpecialValue::none),
                                ::testing::Values(CPUSpecificParams({nchw}, {nchw}, {}, {}))),
                        ConvertToBooleanCPULayerTest::getTestCaseName);

}  // namespace
}  // namespace Conversion
}  // namespace test
}  // namespace ov