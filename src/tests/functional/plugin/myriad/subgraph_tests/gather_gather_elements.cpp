// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dsr_tests_common.hpp"

#include <ngraph/opsets/opset6.hpp>

namespace {

using namespace LayerTestsUtils::vpu;

class Gather_GatherElements : public testing::WithParamInterface<std::string>,
                              public DSR_TestsCommon {
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        SetRefMode(LayerTestsUtils::RefMode::INTERPRETER);

        targetDevice = GetParam();

        const auto imageParam = createParameter(ngraph::element::f32, ngraph::Shape{1, 3, 800, 1216});
        const auto imageGatherIndices = createInputSubgraphWithDSR(ngraph::element::i32, DataShapeWithUpperBound{{300, 64}, {300, 64}});
        const auto imageGather = std::make_shared<ngraph::opset6::Gather>(
            imageParam,
            imageGatherIndices,
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{}, {2}));
        const auto squeeze = std::make_shared<ngraph::opset6::Squeeze>(
            imageGather,
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {0}));
        const auto transpose = std::make_shared<ngraph::opset6::Transpose>(
                squeeze,
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{4}, {1, 0, 2, 3}));

        const auto shapeOf1 = std::make_shared<ngraph::opset6::ShapeOf>(transpose, ngraph::element::i32);
        const auto shapeOf2 = std::make_shared<ngraph::opset6::ShapeOf>(transpose, ngraph::element::i32);
        const auto shapeOf3 = std::make_shared<ngraph::opset6::ShapeOf>(transpose, ngraph::element::i32);
        const auto shapeOf4 = std::make_shared<ngraph::opset6::ShapeOf>(transpose, ngraph::element::i32);
        const auto shapeOf5 = std::make_shared<ngraph::opset6::ShapeOf>(transpose, ngraph::element::i32);
        const auto shapeOf6 = std::make_shared<ngraph::opset6::ShapeOf>(transpose, ngraph::element::i32);

        const auto gatherShape1 = std::make_shared<ngraph::opset6::Gather>(
            shapeOf1,
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0}),
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0}));
        const auto gatherShape2 = std::make_shared<ngraph::opset6::Gather>(
            shapeOf2,
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{}, {1}),
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0}));
        const auto gatherShape3 = std::make_shared<ngraph::opset6::Gather>(
            shapeOf3,
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{}, {2}),
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0}));
        const auto gatherShape4 = std::make_shared<ngraph::opset6::Gather>(
            shapeOf4,
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0}),
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0}));
        const auto gatherShape5 = std::make_shared<ngraph::opset6::Gather>(
            shapeOf5,
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{}, {1}),
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0}));
        const auto gatherShape6 = std::make_shared<ngraph::opset6::Gather>(
            shapeOf6,
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{}, {2}),
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0}));

        const auto unsqueeze1 = std::make_shared<ngraph::opset6::Unsqueeze>(
            gatherShape1,
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {0}));
        const auto unsqueeze2 = std::make_shared<ngraph::opset6::Unsqueeze>(
            gatherShape2,
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {0}));
        const auto unsqueeze3 = std::make_shared<ngraph::opset6::Unsqueeze>(
            gatherShape3,
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {0}));
        const auto unsqueeze4 = std::make_shared<ngraph::opset6::Unsqueeze>(
            gatherShape4,
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {0}));
        const auto unsqueeze5 = std::make_shared<ngraph::opset6::Unsqueeze>(
            gatherShape5,
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {0}));
        const auto unsqueeze6 = std::make_shared<ngraph::opset6::Unsqueeze>(
            gatherShape6,
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {0}));

        const auto concat1 = std::make_shared<ngraph::opset6::Concat>(
            ngraph::NodeVector{unsqueeze1, unsqueeze2, unsqueeze3, ngraph::opset6::Constant::create(unsqueeze1->get_element_type(), ngraph::Shape{1}, {64})},
            0);
        const auto concat2 = std::make_shared<ngraph::opset6::Concat>(
            ngraph::NodeVector{unsqueeze4, unsqueeze5, unsqueeze6, ngraph::opset6::Constant::create(unsqueeze1->get_element_type(), ngraph::Shape{1}, {64})},
            0);

        const auto reshape1 = std::make_shared<ngraph::opset6::Reshape>(
            concat1,
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {-1}),
            false);
        const auto reshape2 = std::make_shared<ngraph::opset6::Reshape>(
            concat2,
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {-1}),
            false);

        const auto equal1 = std::make_shared<ngraph::opset6::Equal>(
            reshape1,
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{4}, {-1, -1, -1, -1}));
        const auto equal2 = std::make_shared<ngraph::opset6::Equal>(
            reshape2,
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{4}, {-1, -1, -1, -1}));

        const auto select1 = std::make_shared<ngraph::opset6::Select>(
            equal1,
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{4}, {1, 1, 1, 1}),
            reshape1);
        const auto select2 = std::make_shared<ngraph::opset6::Select>(
            equal2,
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{4}, {1, 1, 1, 1}),
            reshape2);

        const auto broadcastParam1 = createInputSubgraphWithDSR(ngraph::element::i32, DataShapeWithUpperBound{{300, 1, 1, 64}, {300, 1, 1, 64}});
        const auto broadcast1 = std::make_shared<ngraph::opset6::Broadcast>(broadcastParam1, select1);

        const auto broadcastParam2 = createInputSubgraphWithDSR(ngraph::element::i32, DataShapeWithUpperBound{{300, 1, 1, 64}, {300, 1, 1, 64}});
        const auto broadcast2 = std::make_shared<ngraph::opset6::Broadcast>(broadcastParam2, select2);

        const auto gatherElements1 = std::make_shared<ngraph::opset6::GatherElements>(transpose, broadcast1, 3);
        const auto gatherElements2 = std::make_shared<ngraph::opset6::GatherElements>(transpose, broadcast2, 3);

        m_additionalResults.push_back(std::make_shared<ngraph::opset6::Result>(gatherElements1));
        return gatherElements2;
    }
};

TEST_P(Gather_GatherElements, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_Gather_GatherElements, Gather_GatherElements, testing::Values(CommonTestUtils::DEVICE_MYRIAD));

} // namespace