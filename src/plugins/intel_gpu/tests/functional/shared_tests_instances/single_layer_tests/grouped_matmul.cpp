// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/grouped_matmul.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "openvino/runtime/exec_model_info.hpp"

namespace {
using ov::test::GroupedMatMulCompressedLayerTest;
using ov::test::GroupedMatMulCompressedParams;
using ov::test::GroupedMatMulLayerTest;
using ov::test::GroupedMatMulShapeParams;
using ov::test::TokensPerExpert;
using ov::test::utils::DecompressionType;

// GPU-specific subclass that validates weights precision is preserved in the compiled model.
class GroupedMatMulCompressedLayerTest_GPU : public GroupedMatMulCompressedLayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GroupedMatMulCompressedParams>& obj) {
        return GroupedMatMulCompressedLayerTest::getTestCaseName(obj);
    }

protected:
    void check_results() {
        const auto& test_param = GetParam();
        const ov::element::Type expected_weights_prec = std::get<2>(test_param);
        const auto subtract_type = std::get<6>(test_param);
        const bool has_zero_point = (subtract_type != DecompressionType::empty);

        bool found = false;
        for (const auto& n : compiledModel.get_runtime_model()->get_ordered_ops()) {
            auto it = n->get_rt_info().find("weights_precision");
            if (it != n->get_rt_info().end()) {
                found = true;
                auto actual_prec = ov::element::Type(it->second.as<std::string>());
                ASSERT_EQ(actual_prec, expected_weights_prec)
                    << "Node '" << n->get_friendly_name()
                    << "' has weights_precision=" << actual_prec
                    << " but expected " << expected_weights_prec;

                if (has_zero_point) {
                    auto zp_it = n->get_rt_info().find("wzp_precision");
                    ASSERT_TRUE(zp_it != n->get_rt_info().end())
                        << "Node '" << n->get_friendly_name()
                        << "' is missing 'wzp_precision' rt_info";
                    auto actual_zp_prec = ov::element::Type(zp_it->second.as<std::string>());
                    ASSERT_EQ(actual_zp_prec, expected_weights_prec)
                        << "Node '" << n->get_friendly_name()
                        << "' has wzp_precision=" << actual_zp_prec
                        << " but expected " << expected_weights_prec;
                }
            }
        }
        ASSERT_TRUE(found) << "No node with 'weights_precision' rt_info found in the runtime model";
    }
};

TEST_P(GroupedMatMulCompressedLayerTest_GPU, Inference) {
    run();
    check_results();
}

const std::vector<GroupedMatMulShapeParams> shapes_3d_3d = {
    // 3D x 3D: A:[G,M,K] x B:[G,N,K] -> [G,M,N], dynamic M dim.
    // The GPU plugin lowers this case to a fully_connected (see ops/grouped_matmul.cpp),
    {{ov::PartialShape{4, -1, 128}, {{4, 8, 128}, {4, 1, 128}, {4, 16, 128}}}, {4, 256, 128}, {}},
    {{ov::PartialShape{8, -1, 256}, {{8, 4, 256}, {8, 1, 256}}}, {8, 512, 256}, {}},
};

const std::vector<GroupedMatMulShapeParams> shapes_2d_3d = {
    // 2D x 3D: A:[T,K] x B:[G,N,K] -> [T,N], dynamic T dim.
    {{ov::PartialShape{-1, 128}, {{16, 128}, {32, 128}, {8, 128}}}, {4, 256, 128}, TokensPerExpert{{8, 0, 8, 0}, {0, 16, 0, 16}, {4, 4, 0, 0}}},
    {{ov::PartialShape{-1, 128}, {{12, 128}, {20, 128}}}, {4, 256, 128}, TokensPerExpert{{4, 5, 3, 0}, {0, 7, 6, 7}}},
    {{ov::PartialShape{-1, 256}, {{16, 256}, {32, 256}}}, {8, 512, 256}, TokensPerExpert{{4, 2, 2, 2, 2, 2, 1, 1}, {2, 6, 4, 4, 4, 4, 4, 4}}},
};

const std::vector<ov::element::Type> weights_precisions = {ov::element::u8, ov::element::u4};
const std::vector<ov::element::Type> decompression_precisions = {ov::element::f16};
const std::vector<DecompressionType> subtract_types = {DecompressionType::full, DecompressionType::empty};

INSTANTIATE_TEST_SUITE_P(smoke_GroupedMatMul_f16_2d3d,
                         GroupedMatMulLayerTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_2d_3d),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::Values("GroupedMatMul")),
                         GroupedMatMulLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupedMatMul_f16_3d3d,
                         GroupedMatMulLayerTest,
                         ::testing::Combine(::testing::ValuesIn(shapes_3d_3d),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::Values("GroupedMatMul")),
                         GroupedMatMulLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupedMatMul_Compressed_2d3d,
                         GroupedMatMulCompressedLayerTest_GPU,
                         ::testing::Combine(::testing::ValuesIn(shapes_2d_3d),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::ValuesIn(subtract_types),
                                            ::testing::Values(true),
                                            ::testing::Values(-1, 128),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::Values("GroupedMatMulCompressed"),
                                            ::testing::Values(ov::AnyMap{})),
                         GroupedMatMulCompressedLayerTest_GPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupedMatMul_Compressed_3d3d,
                         GroupedMatMulCompressedLayerTest_GPU,
                         ::testing::Combine(::testing::ValuesIn(shapes_3d_3d),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::ValuesIn(weights_precisions),
                                            ::testing::ValuesIn(decompression_precisions),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::ValuesIn(subtract_types),
                                            ::testing::Values(true),
                                            ::testing::Values(-1, 128),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::Values("GroupedMatMulCompressed"),
                                            ::testing::Values(ov::AnyMap{})),
                         GroupedMatMulCompressedLayerTest_GPU::getTestCaseName);
}  // namespace
