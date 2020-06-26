// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dsr_tests_common.hpp"

#include <functional_test_utils/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

namespace {

using namespace LayerTestsUtils::vpu;

enum DYNAMISM_MODE {
    BOTH_INPUTS_DYNAMIC,
    A_INPUT_DYNAMIC,
    B_INPUT_DYNAMIC
};

struct MatMul_input_setup {
    ngraph::Shape realShape, upperBoundShape;
    bool transpose;
};

struct MatMulTestCase {
    MatMul_input_setup A, B;
};

const auto combinations = testing::Combine(
        testing::Values(
                DYNAMISM_MODE::BOTH_INPUTS_DYNAMIC,
                DYNAMISM_MODE::A_INPUT_DYNAMIC,
                DYNAMISM_MODE::B_INPUT_DYNAMIC),
        testing::Values(
                ngraph::element::f16,
                ngraph::element::f32),
        testing::Values(
// JIRA: 33925           MatMulTestCase{{{1024}, false, 1, {}, {}, 0}, {{1024, 1000}, false, 0, {}, {}, 1}},
// JIRA: 33925           MatMulTestCase{{{1024}, true, 1, {1, 0}, {}, 0}, {{1, 1000}, false, 0, {}, {}, 1}},
        MatMulTestCase{{{3, 10, 1024}, {5, 10, 1024}, false},
                       {{1024, 800}, {1024, 1000}, false}},
        MatMulTestCase{{{2, 10, 1024}, {5, 10, 1024}, false},
                       {{1, 1024, 500}, {1, 1024, 1000}, false}},
        MatMulTestCase{{{1, 10, 1024}, {5, 10, 1024}, false},
                       {{1, 800, 1024}, {1, 1000, 1024}, true}},
        MatMulTestCase{{{3, 10, 1024}, {3, 10, 1024}, false},
                       {{2, 1, 1000, 1024}, {5, 1, 1000, 1024}, true}}),
        testing::Values(CommonTestUtils::DEVICE_MYRIAD));


using DataType = ngraph::element::Type_t;
using DataDims = ngraph::Shape;

using Parameters = std::tuple<
    DYNAMISM_MODE,
    DataType,
    MatMulTestCase,
    LayerTestsUtils::TargetDevice
>;

class DSR_MatMul : public testing::WithParamInterface<Parameters>,
                   public DSR_TestsCommon {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& params = GetParam();
        const auto& mode = std::get<0>(params);
        const auto& data_type = std::get<1>(params);
        const auto& matmul_setup = std::get<2>(params);
        targetDevice = std::get<3>(params);

        std::shared_ptr<ngraph::Node> inputA, inputB;

        switch (mode) {
            case DYNAMISM_MODE::BOTH_INPUTS_DYNAMIC: {
                inputA = createInputSubgraphWithDSR(data_type, DataShapeWithUpperBound{matmul_setup.A.realShape, matmul_setup.A.upperBoundShape});
                inputB = createInputSubgraphWithDSR(data_type, DataShapeWithUpperBound{matmul_setup.B.realShape, matmul_setup.B.upperBoundShape});
                break;
            }
            case DYNAMISM_MODE::A_INPUT_DYNAMIC: {
                inputA = createInputSubgraphWithDSR(data_type, DataShapeWithUpperBound{matmul_setup.A.realShape, matmul_setup.A.upperBoundShape});
                const auto inputBParam = std::make_shared<ngraph::opset3::Parameter>(data_type, matmul_setup.B.realShape);
                m_parameterVector.push_back(inputBParam);
                inputB = inputBParam;
                break;
            }
            case DYNAMISM_MODE::B_INPUT_DYNAMIC: {
                const auto inputAParam = std::make_shared<ngraph::opset3::Parameter>(data_type, matmul_setup.A.realShape);
                m_parameterVector.push_back(inputAParam);
                inputA = inputAParam;
                inputB = createInputSubgraphWithDSR(data_type, DataShapeWithUpperBound{matmul_setup.B.realShape, matmul_setup.B.upperBoundShape});
                break;
            }
            default:
                NGRAPH_UNREACHABLE("UNKNOWN DYNAMISM MODE for MatMul DSR graph comparison test");
        }

        const auto matMul = std::make_shared<ngraph::opset3::MatMul>(inputA, inputB, matmul_setup.A.transpose, matmul_setup.B.transpose);

        return matMul;
    }
};

TEST_P(DSR_MatMul, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_CASE_P(DynamicMatMul, DSR_MatMul, combinations);
}  // namespace
