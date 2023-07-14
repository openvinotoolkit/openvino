// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/fusing_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/rt_info/decompression.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace SubgraphTestsDefinitions {

/* This test checks that the ConvertMatMulToFC transformation should work and the MatMul node is converted to the FC node.
 * The Convert node should be removed on the CPU plugin side.

 * Graph before:
   ------------             ------------
   |Input(f32)|             |Input(f16)|
   ------------             ------------
        |                        |
        |         ---------------------------------
        |         |Convert(decompression f16->f32)|
        |         ---------------------------------
        |                        |
    -----------------------------------------------
    |                   MatMul                    |
    -----------------------------------------------
                          |
                       --------
                       |Output|
                       --------

 * Exec graph:
   ------------    ------------
   |Input(f32)|    |Input(f16)|
   ------------    ------------
        |               |
   ----------------------------
   |      FullyConnected      |
   ----------------------------
                 |
              --------
              |Output|
              --------
*/

using MatMulDecompressConvertParams = std::tuple<
    std::vector<InputShape>, // input shapes
    std::pair<bool, bool>    // transposeA, transposeB
>;

class MatMulDecompressConvertTest : public testing::WithParamInterface<MatMulDecompressConvertParams>,
                                    virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MatMulDecompressConvertParams> obj) {
        std::vector<InputShape> inputShapes;
        std::pair<bool, bool> transpose;

        std::tie(inputShapes, transpose) = obj.param;

        std::ostringstream result;
        for (const auto& shape : inputShapes) {
            result << CommonTestUtils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                auto itr = shape.second.begin();
                do {
                    result << CommonTestUtils::vec2str(*itr);
                } while (++itr != shape.second.end() && result << "_");
            }
            result << ")_";
        }
        result << "transpose_a=" << transpose.first << "_";
        result << "transpose_b=" << transpose.second << "_";

        return result.str();
    }

protected:
    template<typename T>
    void transposeShape(T& shape) {
        IE_ASSERT(shape.size() > 1);
        std::swap(*(shape.end() - 1), *(shape.end() - 2));
    }

    void SetUp() override {
        abs_threshold = 0.01;
        targetDevice = CommonTestUtils::DEVICE_CPU;

        std::vector<InputShape> inputShapes;
        std::pair<bool, bool> transpose;

        std::tie(inputShapes, transpose) = this->GetParam();

        init_input_shapes(inputShapes);

        bool transpA = transpose.first;
        bool transpB = transpose.second;

        if (transpA) {
            transposeShape(inputDynamicShapes[0]);
            for (auto& shapes : targetStaticShapes) {
                transposeShape(shapes[0]);
            }
        }
        if (transpB) {
            transposeShape(inputDynamicShapes[1]);
            for (auto& shapes : targetStaticShapes) {
                transposeShape(shapes[1]);
            }
        }

        const auto& inShapeA = inputDynamicShapes[0];
        const auto& inShapeB = inputDynamicShapes[1];

        std::string cpuNodeType = "FullyConnected";

        auto params = builder::makeDynamicParams(element::f32, {inShapeA});
        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<opset1::Parameter>(params));

        auto matrixB = ngraph::builder::makeConstant<float16>(element::f16, inShapeB.get_shape(), {}, true);
        auto convert = std::make_shared<ngraph::opset1::Convert>(matrixB, element::f32);
        mark_as_decompression(convert);
        auto matMul = builder::makeMatMul(paramOuts[0], convert, transpA, transpB);

        function = CPUTestsBase::makeNgraphFunction(element::f32, params, matMul, cpuNodeType);
    }
};

TEST_P(MatMulDecompressConvertTest, CompareWithRefs) {
    run();
    CheckNumberOfNodesWithType(compiledModel, "FullyConnected", 1);
    CheckNumberOfNodesWithType(compiledModel, "Convert", 0);
    CheckNumberOfNodesWithType(compiledModel, "Reorder", 0);
}

namespace {

const std::vector<MatMulDecompressConvertParams> IS2D_smoke = {
    {static_shapes_to_test_representation({{2, 3}, {3, 4}}), {false, false}},
    {static_shapes_to_test_representation({{2, 3}, {3, 4}}), {false, true}},
    {static_shapes_to_test_representation({{2, 3}, {3, 4}}), {true, false}},
    {static_shapes_to_test_representation({{2, 3}, {3, 4}}), {true, true}},
    {static_shapes_to_test_representation({{40, 60}, {60, 80}}), {false, true}},
};

INSTANTIATE_TEST_SUITE_P(smoke_FC_2D, MatMulDecompressConvertTest, ::testing::ValuesIn(IS2D_smoke),
                         MatMulDecompressConvertTest::getTestCaseName);

const std::vector<MatMulDecompressConvertParams> IS3D_smoke = {
    {static_shapes_to_test_representation({{1, 2, 3}, {3, 4}}), {false, false}},
    {static_shapes_to_test_representation({{1, 2, 3}, {3, 4}}), {false, true}},
    {static_shapes_to_test_representation({{1, 2, 3}, {3, 4}}), {true, false}},
    {static_shapes_to_test_representation({{1, 2, 3}, {3, 4}}), {true, true}},
    {static_shapes_to_test_representation({{2, 3}, {1, 3, 4}}), {false, false}},
    {static_shapes_to_test_representation({{2, 3}, {1, 3, 4}}), {false, true}},
    {static_shapes_to_test_representation({{2, 3}, {1, 3, 4}}), {true, false}},
    {static_shapes_to_test_representation({{2, 3}, {1, 3, 4}}), {true, true}},
};

INSTANTIATE_TEST_SUITE_P(smoke_FC_3D, MatMulDecompressConvertTest, ::testing::ValuesIn(IS3D_smoke),
                         MatMulDecompressConvertTest::getTestCaseName);

} // namespace

} // namespace SubgraphTestsDefinitions
