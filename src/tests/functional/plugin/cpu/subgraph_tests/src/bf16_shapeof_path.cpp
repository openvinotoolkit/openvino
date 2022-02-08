// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/base/ov_subgraph.hpp>
#include <ngraph_functions/builders.hpp>
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace ov::test;
using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

class BF16ShapeOfPath : virtual public SubgraphBaseTest, public CPUTestsBase {
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;

        if (InferenceEngine::with_cpu_x86_avx512f()) {
            configuration[InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16] = InferenceEngine::PluginConfigParams::YES;
        }

        std::vector <InputShape> inputShapes{
            {{-1, -1, -1, -1}, {{{1, 11, 4, 4}, {2, 7, 6, 5}}}},
            {{-1, -1}, {{1, 2}, {2, 2}}}
        };

        init_input_shapes({inputShapes});

        auto ngPrc = ngraph::element::f32;
        auto inputParams = ngraph::builder::makeDynamicParams(ngPrc, inputDynamicShapes);

        // shape_of subgraph
        auto shapeof = std::make_shared<ngraph::opset3::ShapeOf>(inputParams[1], ngraph::element::i32);
        auto convert = std::make_shared<ngraph::opset3::Convert>(shapeof, ngraph::element::f32);

        auto sndInAdd = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::f32, ov::Shape{1}, std::vector<float>{1});
        auto add = std::make_shared<ngraph::opset3::Add>(convert, sndInAdd);

        auto sndInMult = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::f32, ov::Shape{1}, std::vector<float>{2});
        auto mult = std::make_shared<ngraph::opset3::Multiply>(add, sndInMult);

        // interpolate node
        using InterpOp = ngraph::op::v4::Interpolate;
        InterpOp::InterpolateAttrs interpAttr{InterpOp::InterpolateMode::NEAREST,
                                              InterpOp::ShapeCalcMode::SCALES,
                                              std::vector<size_t>{0, 0, 0, 0},
                                              std::vector<size_t>{0, 0, 0, 0},
                                              InterpOp::CoordinateTransformMode::ASYMMETRIC,
                                              InterpOp::NearestMode::SIMPLE,
                                              false,
                                              -0.75f};
        std::vector<int32_t> sizes{20, 20};
        auto sizesInput = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i32, ov::Shape{sizes.size()}, sizes);
        auto axesInput = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i32, ov::Shape{2}, std::vector<int32_t>{2, 3});

        auto interpolate = std::make_shared<InterpOp>(inputParams[0],
                                                      sizesInput,
                                                      mult,
                                                      axesInput,
                                                      interpAttr);

        function = makeNgraphFunction(ngPrc, inputParams, interpolate, "InterpolateShapeOfPath");
    }
};

TEST_F(BF16ShapeOfPath, smoke) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
    CheckNodeRuntimePrecision(compiledModel, "Eltwise", InferenceEngine::Precision::FP32);
}

} // namespace SubgraphTestsDefinitions