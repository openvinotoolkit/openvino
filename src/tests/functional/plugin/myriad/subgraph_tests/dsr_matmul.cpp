// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dsr_tests_common.hpp"

#include <shared_test_classes/base/layer_test_utils.hpp>
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
        MatMulTestCase{{{3, 10, 128}, {5, 10, 128}, false},
                       {{128, 80}, {128, 100}, false}},
        MatMulTestCase{{{2, 10, 128}, {5, 10, 128}, false},
                       {{1, 128, 50}, {1, 128, 100}, false}},
        MatMulTestCase{{{1, 10, 128}, {5, 10, 128}, false},
                       {{1, 80, 128}, {1, 100, 128}, true}},
        MatMulTestCase{{{3, 10, 128}, {3, 10, 128}, false},
                       {{2, 1, 100, 128}, {5, 1, 100, 128}, true}}),
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
                inputB = createParameter(data_type, matmul_setup.B.realShape);
                break;
            }
            case DYNAMISM_MODE::B_INPUT_DYNAMIC: {
                inputA = createParameter(data_type, matmul_setup.A.realShape);
                inputB = createInputSubgraphWithDSR(data_type, DataShapeWithUpperBound{matmul_setup.B.realShape, matmul_setup.B.upperBoundShape});
                break;
            }
            default:
                NGRAPH_UNREACHABLE("UNKNOWN DYNAMISM MODE for MatMul DSR graph comparison test");
        }

        return std::make_shared<ngraph::opset3::MatMul>(inputA, inputB, matmul_setup.A.transpose, matmul_setup.B.transpose);
    }
};

TEST_P(DSR_MatMul, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_DynamicMatMul, DSR_MatMul, combinations);
}  // namespace
