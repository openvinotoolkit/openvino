// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/grouped_matmul.hpp"
#include "openvino/op/parameter.hpp"

namespace {

using ov::test::InputShape;

// Params: input shapes for {mat_a, mat_b, offsets}, offsets values (size G, sum == total tokens),
//         activation precision, offsets precision, target device.
using GroupedMatMulGpuTestParams = std::tuple<std::vector<InputShape>,
                                              std::vector<int32_t>,
                                              ov::element::Type,
                                              ov::element::Type,
                                              std::string>;

class GroupedMatMulGpuTest : public testing::WithParamInterface<GroupedMatMulGpuTestParams>,
                             virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GroupedMatMulGpuTestParams>& obj) {
        const auto& [input_shapes, offsets, act_prec, off_prec, target_device] = obj.param;
        std::ostringstream result;
        result << "IS=(";
        for (size_t i = 0; i < input_shapes.size(); ++i) {
            result << ov::test::utils::partialShape2str({input_shapes[i].first})
                   << (i < input_shapes.size() - 1 ? "_" : "");
        }
        result << ")_TS=";
        for (size_t i = 0; i < input_shapes.front().second.size(); ++i) {
            result << "{";
            for (size_t j = 0; j < input_shapes.size(); ++j) {
                result << ov::test::utils::vec2str(input_shapes[j].second[i])
                       << (j < input_shapes.size() - 1 ? "_" : "");
            }
            result << "}_";
        }
        result << "offsets=" << ov::test::utils::vec2str(offsets) << "_";
        result << "actPrec=" << act_prec << "_";
        result << "offPrec=" << off_prec << "_";
        result << "device=" << target_device;
        return result.str();
    }

protected:
    std::vector<int32_t> m_offsets;
    ov::element::Type m_off_prec;

    void SetUp() override {
        const auto& [input_shapes, offsets, act_prec, off_prec, target_device] = GetParam();
        targetDevice = target_device;
        m_offsets = offsets;
        m_off_prec = off_prec;

        init_input_shapes(input_shapes);

        auto mat_a = std::make_shared<ov::op::v0::Parameter>(act_prec, inputDynamicShapes[0]);
        auto mat_b = std::make_shared<ov::op::v0::Parameter>(act_prec, inputDynamicShapes[1]);
        auto offsets_param = std::make_shared<ov::op::v0::Parameter>(off_prec, inputDynamicShapes[2]);
        mat_a->set_friendly_name("mat_a");
        mat_b->set_friendly_name("mat_b");
        offsets_param->set_friendly_name("offsets");

        auto gmm = std::make_shared<ov::op::v17::GroupedMatMul>(mat_a, mat_b, offsets_param);
        function = std::make_shared<ov::Model>(ov::OutputVector{gmm},
                                               ov::ParameterVector{mat_a, mat_b, offsets_param});
    }

    void generate_inputs(const std::vector<ov::Shape>& target_shapes) override {
        inputs.clear();
        const auto& func_inputs = function->inputs();

        // mat_a: random within [-1, 1] range.
        {
            const auto& node = func_inputs[0];
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = -1;
            in_data.range = 2;
            in_data.resolution = 32;
            auto tensor = ov::test::utils::create_and_fill_tensor(node.get_element_type(), target_shapes[0], in_data);
            inputs.insert({node.get_node_shared_ptr(), tensor});
        }
        // mat_b: random within [-1, 1] range.
        {
            const auto& node = func_inputs[1];
            ov::test::utils::InputGenerateData in_data;
            in_data.start_from = -1;
            in_data.range = 2;
            in_data.resolution = 32;
            auto tensor = ov::test::utils::create_and_fill_tensor(node.get_element_type(), target_shapes[1], in_data);
            inputs.insert({node.get_node_shared_ptr(), tensor});
        }
        // offsets: user-supplied values, cast to requested integer precision.
        {
            const auto& node = func_inputs[2];
            ov::Tensor tensor(node.get_element_type(), target_shapes[2]);
            if (m_off_prec == ov::element::i32) {
                auto* ptr = tensor.data<int32_t>();
                for (size_t i = 0; i < m_offsets.size(); ++i) {
                    ptr[i] = m_offsets[i];
                }
            } else if (m_off_prec == ov::element::i64) {
                auto* ptr = tensor.data<int64_t>();
                for (size_t i = 0; i < m_offsets.size(); ++i) {
                    ptr[i] = static_cast<int64_t>(m_offsets[i]);
                }
            } else {
                FAIL() << "Unsupported offsets precision: " << m_off_prec;
            }
            inputs.insert({node.get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(GroupedMatMulGpuTest, Inference) {
    // Loose tolerance for f16 accumulation over K.
    abs_threshold = 0.05f;
    rel_threshold = 0.05f;
    run();
}

// 2D x 3D case with 3 experts (one empty), K=128, N=64.
// mat_a and offsets have dynamic first dim so the new-shape-infer path is exercised on GPU.
INSTANTIATE_TEST_SUITE_P(
    smoke_GroupedMatMul_2d_3d,
    GroupedMatMulGpuTest,
    ::testing::Combine(
        ::testing::Values(std::vector<InputShape>{
            {{-1, 128}, {ov::Shape{10, 128}}},         // mat_a: [T, K]
            {{3, 64, 128}, {ov::Shape{3, 64, 128}}},   // mat_b: [G, N, K] (static)
            {{-1}, {ov::Shape{3}}}}),                  // offsets: [G]
        ::testing::Values(std::vector<int32_t>{4, 0, 6}),
        ::testing::Values(ov::element::f16),
        ::testing::Values(ov::element::i32),
        ::testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
    GroupedMatMulGpuTest::getTestCaseName);

}  // namespace
