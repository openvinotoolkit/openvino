// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

/*
The main purpose of the tests is to test cyclic inplace resolution in order to make sure that output edges are referenced whenever possible.
*/

using namespace CPUTestUtils;
namespace ov {
namespace test {

using VectorShapes = std::vector<InputShape>;

class InplaceResolveIOTestBase : public testing::WithParamInterface<VectorShapes>,
                                 virtual public ov::test::SubgraphBaseTest,
                                 public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<VectorShapes> obj) {
        VectorShapes& inputShapes = obj.param;

        std::ostringstream result;
        result << "IS=";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                for (const auto& itr : shape.second) {
                    result << ov::test::utils::vec2str(itr);
                }
            }
            result << ")";
        }
        return result.str();
    }

protected:
    const ov::element::Type precision = ov::element::f32;
};

class RNNConcatSubgraphTest : public InplaceResolveIOTestBase {
/*This test runs the following subgraph:

                      H_t      X  seq_lens
                     /   \     |     /
                    /     \    |    / 
                 Softmax0  RNNSequence
                    \       /(Ho)   \(Y)
                     \     /         \
                     Concat       Reshape1
                       |              |
                       |              |
                     Result0       Result1

Edge Concat -> Result0 can share memory of inference output; Reshape1 -> Result1 can share memory of inference output;
*/
public:
    void SetUp() override {
        constexpr size_t softmax_axis = 1ul;
        constexpr size_t hidden_size = 3;
        targetDevice = ov::test::utils::DEVICE_CPU;
        auto& input_shape = this->GetParam();
        init_input_shapes({input_shape});
        ov::ParameterVector params;
        params.push_back(std::make_shared<ov::op::v0::Parameter>(precision, inputDynamicShapes[0]));
        params.push_back(std::make_shared<ov::op::v0::Parameter>(precision, inputDynamicShapes[1]));
        params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::i32, inputDynamicShapes[2]));
        auto soft_max = std::make_shared<ov::op::v1::Softmax>(params[1], softmax_axis);

        std::vector<float> W(hidden_size * 3 * 8);
        std::iota(W.begin(), W.end(), 0.2);
        std::vector<float> R(hidden_size * 3 * hidden_size);
        std::iota(R.begin(), R.end(), 0.5);
        std::vector<float> B(hidden_size * 3);
        std::iota(B.begin(), B.end(), 0.5);
        auto rnnseq = std::make_shared<ov::op::v5::GRUSequence>(
                params[0],   // X
                params[1],   // H_t
                params[2],   // sequence_lengths
                ov::op::v0::Constant::create(precision, {1, hidden_size * 3, 8}, W),  // W [num_directions, 3 * hidden_size, input_size]
                ov::op::v0::Constant::create(precision, {1, hidden_size * 3, hidden_size}, R),  // R [num_directions, 3 * hidden_size, hidden_size]
                ov::op::v0::Constant::create(precision, {1, hidden_size * 3}, B),  // B [num_directions, 3 * hidden_size]
                hidden_size, op::RecurrentSequenceDirection::FORWARD);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{rnnseq->outputs()[1], soft_max}, 0);
        auto result_0 = std::make_shared<ov::op::v0::Result>(concat);  // Ho [batch_size, num_directions, hidden_size]

        auto reshape_param_1 = ov::op::v0::Constant::create(ov::element::i32, {1}, {1});
        auto reshape_1 = std::make_shared<ov::op::v0::Squeeze>(rnnseq->outputs()[0], reshape_param_1);  // Y [batch_size, num_directions, seq_len, hidden_size]
        auto result_1 = std::make_shared<ov::op::v0::Result>(reshape_1);

        function = std::make_shared<ov::Model>(ov::ResultVector{result_0, result_1}, params, "Subgraph0");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0lu; i < funcInputs.size(); i++) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (i == 2) {  // sequence_lengths
                tensor = ov::Tensor{ov::element::i32, targetInputStaticShapes[i]};
                auto data = tensor.data<ov::element_type_traits<ov::element::i32>::value_type>();
                for (size_t j = 0lu; j < ov::shape_size(targetInputStaticShapes[i]); j++) {
                    data[j] = targetInputStaticShapes[0].at(1);
                }
            } else {
                ov::test::utils::InputGenerateData in_data;
                in_data.start_from = 0;
                in_data.range = 10;
                in_data.resolution = 1000;
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(RNNConcatSubgraphTest, smoke_CompareWithRefs) {
    run();
}

namespace {

const std::vector<std::vector<InputShape>> inputShapes2 = {
    {
        // static
        {{2, 10, 8}, {{2, 10, 8}}}, // X [batch_size, seq_length, input_size]
        {{2, 1, 3}, {{2, 1, 3}}},   // H_t [batch_size, num_directions, hidden_size]
        {{2}, {{2}}}                // [batch_size]
    },
    {
        // dynamic batch_size
        {{-1, 10, 8}, {{1, 10, 8}, {2, 10, 8}}}, // param0 [batch_size, seq_length, input_size]
        {{-1, 1, 3}, {{1, 1, 3}, {2, 1, 3}}}, // param1 [batch_size, num_directions, hidden_size]
        {{-1}, {{1}, {2}}},  // param2 [batch_size]
    },
    {
        // dynamic seq_length
        {{-1, -1, 8}, {{2, 10, 8}, {1, 5, 8}}}, // param0 [batch_size, seq_length, input_size]
        {{-1, 1, 3}, {{2, 1, 3}, {1, 1, 3}}}, // param1 [batch_size, num_directions, hidden_size]
        {{-1}, {{2}, {1}}},  // param2 [batch_size]
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_inplace_resolve_io, RNNConcatSubgraphTest,
                        ::testing::ValuesIn(inputShapes2),
                        RNNConcatSubgraphTest::getTestCaseName);
}  // namespace

class SoftmaxAddReshapeOutputSubgraphTest : public InplaceResolveIOTestBase {
/*This test runs the following subgraph:

                      param
                        |
                        |
                      Softmax
                     /     \
                    /       \
                   Add     Reshape0
                    |         |
                    |         |
                 Result0   Result1

expect edge Reshape0->Result1 to be referenced by its upstreams, instead of referencing to its upstreams.
*/
protected:
    void SetUp() override {
        constexpr size_t softmax_axis = 1ul;
        targetDevice = ov::test::utils::DEVICE_CPU;
        auto& input_shape = this->GetParam();
        init_input_shapes({input_shape});
        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(precision, shape));
        }

        auto soft_max = std::make_shared<ov::op::v1::Softmax>(params.front(), softmax_axis);

        auto add_const = ov::op::v0::Constant::create(precision, {1}, std::vector<float>({1.0f}));
        auto add_0 = ov::test::utils::make_eltwise(soft_max, add_const, ov::test::utils::EltwiseTypes::ADD);
        auto result_0 = std::make_shared<ov::op::v0::Result>(add_0);    // dummy output

        auto reshape_param_0 = ov::op::v0::Constant::create(ov::element::i32, {1}, {0});
        auto reshape_0 = std::make_shared<ov::op::v0::Unsqueeze>(soft_max, reshape_param_0);
        auto result_1 = std::make_shared<ov::op::v0::Result>(reshape_0);    // target output

        function = std::make_shared<ov::Model>(ov::ResultVector{result_0, result_1}, params, "Subgraph1");
    }
};

TEST_P(SoftmaxAddReshapeOutputSubgraphTest, smoke_CompareWithRefs) {
    run();
}


class SoftmaxAddReshapeTwoOutputsSubgraphTest : public InplaceResolveIOTestBase {
/*This test runs the following subgraph:

                      param
                        |
                        |
                      Softmax
                     /       \
                    /         \
                   Add       Reshape0
                    |         |      \
                    |         |       \
                 Result0   Reshape1  Result2
                              |         
                              |
                            Result1

Hope Reshape0 could resolve downstream, so either edge Reshape1 -> Result1 or Reshape0 -> Result2
could get a chance to be referenced by infer request.
*/
public:
    void SetUp() override {
        constexpr size_t softmax_axis = 1ul;
        targetDevice = ov::test::utils::DEVICE_CPU;
        auto& input_shape = this->GetParam();
        init_input_shapes({input_shape});
        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(precision, shape));
        }
        auto soft_max = std::make_shared<ov::op::v1::Softmax>(params.front(), softmax_axis);

        auto add_const = ov::op::v0::Constant::create(precision, {1}, std::vector<float>({1.0f}));
        auto add_0 = ov::test::utils::make_eltwise(soft_max, add_const, ov::test::utils::EltwiseTypes::ADD);
        auto result_0 = std::make_shared<ov::op::v0::Result>(add_0);    // dummy output

        auto reshape_param_0 = ov::op::v0::Constant::create(ov::element::i32, {1}, {0});
        auto reshape_0 = std::make_shared<ov::op::v0::Unsqueeze>(soft_max, reshape_param_0);
        reshape_0->set_friendly_name("reshape_0");
        auto result_2 = std::make_shared<ov::op::v0::Result>(reshape_0);    // dummy output

        auto reshape_param_1 = ov::op::v0::Constant::create(ov::element::i32, {1}, {0});
        auto reshape_1 = std::make_shared<ov::op::v0::Unsqueeze>(reshape_0, reshape_param_1);
        reshape_1->set_friendly_name("reshape_1");
        auto result_1 = std::make_shared<ov::op::v0::Result>(reshape_1);    // target output

        function = std::make_shared<ov::Model>(ov::ResultVector{result_0, result_1, result_2}, params, "Subgraph2");
    }
};

TEST_P(SoftmaxAddReshapeTwoOutputsSubgraphTest, smoke_CompareWithRefs) {
    run();
}

class InputReshapeOutputSubgraphTest : public InplaceResolveIOTestBase {
/*This test runs the following subgraph:

                      param
                        |
                        |
                    Reshape0
                        |
                        |
                     Result0

Edge Reshape0 -> Result0 cannot be referenced by its upstreams as its upstream is an input.
*/
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        auto& input_shape = this->GetParam();
        init_input_shapes({input_shape});
        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(precision, shape));
        }

        auto reshape_param_0 = ov::op::v0::Constant::create(ov::element::i32, {1}, {0});
        auto reshape_0 = std::make_shared<ov::op::v0::Unsqueeze>(params.front(), reshape_param_0);
        auto result_0 = std::make_shared<ov::op::v0::Result>(reshape_0);    // target output

        function = std::make_shared<ov::Model>(ov::ResultVector{result_0}, params, "Subgraph3");
    }
};

TEST_P(InputReshapeOutputSubgraphTest, smoke_CompareWithRefs) {
    run();
}


namespace {

const std::vector<std::vector<InputShape>> inputShapes = {
    {
        // static
        {{2, 64}, {{2, 64}}}
    },
    {
        // dynamic
        {{2, -1}, {{2, 1}, {2, 64}}}
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_inplace_resolve_io, SoftmaxAddReshapeOutputSubgraphTest,
                        ::testing::ValuesIn(inputShapes),
                        SoftmaxAddReshapeOutputSubgraphTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_inplace_resolve_io, SoftmaxAddReshapeTwoOutputsSubgraphTest,
                        ::testing::ValuesIn(inputShapes),
                        SoftmaxAddReshapeTwoOutputsSubgraphTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_inplace_resolve_io, InputReshapeOutputSubgraphTest,
                        ::testing::ValuesIn(inputShapes),
                        InputReshapeOutputSubgraphTest::getTestCaseName);
}  // namespace

}  // namespace test
}  // namespace ov
