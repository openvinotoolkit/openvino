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
    std::pair<bool, bool>,   // transposeA, transposeB
    std::map<std::string, std::string> // additional config
>;

class MatMulDecompressConvertTest : public testing::WithParamInterface<MatMulDecompressConvertParams>,
                                    virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MatMulDecompressConvertParams> obj) {
        std::vector<InputShape> inputShapes;
        std::pair<bool, bool> transpose;
        std::map<std::string, std::string> additionalConfig;

        std::tie(inputShapes, transpose, additionalConfig) = obj.param;

        std::ostringstream result;
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                auto itr = shape.second.begin();
                do {
                    result << ov::test::utils::vec2str(*itr);
                } while (++itr != shape.second.end() && result << "_");
            }
            result << ")_";
        }
        result << "transpose_a=" << transpose.first << "_";
        result << "transpose_b=" << transpose.second << "_";

        result << "config=(";
        for (const auto& configEntry : additionalConfig) {
            result << configEntry.first << ", " << configEntry.second << ":";
        }
        result << ")";

        return result.str();
    }

protected:
    template<typename T>
    void transposeShape(T& shape) {
        IE_ASSERT(shape.size() > 1);
        std::swap(*(shape.end() - 1), *(shape.end() - 2));
    }

    void CheckConstFP16() const {
        auto getExecValue = [](const ov::Node::RTMap& rtInfo, const std::string &paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            IE_ASSERT(rtInfo.end() != it);
            return it->second.as<std::string>();
        };

        const auto execFunction = compiledModel.get_runtime_model();
        ASSERT_NE(nullptr, execFunction);
        for (const auto &fcNode : execFunction->get_ops()) {
            if (getExecValue(fcNode->get_rt_info(), ExecGraphInfoSerialization::LAYER_TYPE) == "FullyConnected") {
                const auto &constNode = fcNode->get_input_node_shared_ptr(1);
                ASSERT_EQ(getExecValue(constNode->get_rt_info(), ExecGraphInfoSerialization::LAYER_TYPE), "Const");
                ASSERT_EQ(getExecValue(constNode->get_rt_info(), ExecGraphInfoSerialization::OUTPUT_PRECISIONS), "FP16");
            }
        }
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        std::vector<InputShape> inputShapes;
        std::pair<bool, bool> transpose;
        std::map<std::string, std::string> additionalConfig;

        std::tie(inputShapes, transpose, additionalConfig) = this->GetParam();

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

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        ElementType netType = element::f32;
        if (additionalConfig[PluginConfigParams::KEY_ENFORCE_BF16] == PluginConfigParams::YES)
            inType = outType = netType = ElementType::bf16;
        else
            inType = outType = netType;

        std::string cpuNodeType = "FullyConnected";

        auto params = builder::makeDynamicParams(inType, {inShapeA});
        auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<opset1::Parameter>(params));

        auto matrixB = ngraph::builder::makeConstant<float16>(element::f16, inShapeB.get_shape(), {}, true);
        auto convert = std::make_shared<ngraph::opset1::Convert>(matrixB, inType);
        mark_as_decompression(convert);
        auto matMul = builder::makeMatMul(paramOuts[0], convert, transpA, transpB);

        function = CPUTestsBase::makeNgraphFunction(netType, params, matMul, cpuNodeType);
    }
};

TEST_P(MatMulDecompressConvertTest, CompareWithRefs) {
    run();
    CheckNumberOfNodesWithType(compiledModel, "FullyConnected", 1);
    CheckNumberOfNodesWithType(compiledModel, "Convert", 0);
    CheckNumberOfNodesWithType(compiledModel, "Reorder", 0);
    CheckConstFP16();
}

namespace {

const std::vector<std::pair<bool, bool>> transposeParams = {
    {false, false},
    {false, true},
    {true, false},
    {true, true},
};

std::vector<std::map<std::string, std::string>> filterAdditionalConfig() {
    std::vector<std::map<std::string, std::string>> additionalConfig;
    additionalConfig.push_back(std::map<std::string, std::string>{/* empty config */});
    if (with_cpu_x86_avx512_core()) {
        additionalConfig.push_back({{PluginConfigParams::KEY_ENFORCE_BF16, PluginConfigParams::YES}});
    }

    return additionalConfig;
}

const auto testParams2D_smoke = ::testing::Combine(
        ::testing::Values(static_shapes_to_test_representation({{2, 3}, {3, 4}})),
        ::testing::ValuesIn(transposeParams),
        ::testing::ValuesIn(filterAdditionalConfig()));

INSTANTIATE_TEST_SUITE_P(smoke_FC_2D, MatMulDecompressConvertTest, testParams2D_smoke,
                         MatMulDecompressConvertTest::getTestCaseName);

const auto testParams3D_smoke = ::testing::Combine(
        ::testing::Values(static_shapes_to_test_representation({{1, 2, 3}, {3, 4}}),
                          static_shapes_to_test_representation({{2, 3}, {1, 3, 4}})),
        ::testing::ValuesIn(transposeParams),
        ::testing::ValuesIn(filterAdditionalConfig()));

INSTANTIATE_TEST_SUITE_P(smoke_FC_3D, MatMulDecompressConvertTest, testParams3D_smoke,
                         MatMulDecompressConvertTest::getTestCaseName);

} // namespace

} // namespace SubgraphTestsDefinitions
