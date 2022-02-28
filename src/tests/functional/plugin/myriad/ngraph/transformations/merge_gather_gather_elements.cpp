// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/ngraph/transformations/merge_gather_gather_elements.hpp>

#include <vpu/ngraph/operations/exp_gather_elements.hpp>
#include <common_test_utils/test_common.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>

#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pass/manager.hpp>

namespace {

using DataType = ngraph::element::Type_t;
using DataDims = ngraph::Shape;

class MergeGatherGatherElements : public CommonTestUtils::TestsCommon {
public:
    void SetUp() override {
        ngraph::helpers::CompareFunctions(*transform(), *reference());
    }

protected:
    std::shared_ptr<const ngraph::Function> transform() const {
        const auto imageParam = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 800, 1216});
        const auto imageGatherIndices = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i32, ngraph::Shape{300, 64});
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

        const auto broadcastParam1 = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i32, ngraph::Shape{300, 1, 1, 64});
        const auto broadcast1 = std::make_shared<ngraph::opset6::Broadcast>(broadcastParam1, select1);

        const auto broadcastParam2 = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i32, ngraph::Shape{300, 1, 1, 64});
        const auto broadcast2 = std::make_shared<ngraph::opset6::Broadcast>(broadcastParam2, select2);

        const auto gatherElements1 = std::make_shared<ngraph::opset6::GatherElements>(transpose, broadcast1, 3);
        const auto gatherElements2 = std::make_shared<ngraph::opset6::GatherElements>(transpose, broadcast2, 3);

        const auto function = std::make_shared<ngraph::Function>(
            ngraph::OutputVector{gatherElements1->output(0), gatherElements2->output(0)},
            ngraph::ParameterVector{imageParam, imageGatherIndices, broadcastParam1, broadcastParam2},
            "GatherGatherElements");

        ngraph::pass::Manager manager;
        manager.register_pass<vpu::MergeGatherGatherElements>();
        manager.run_passes(function);

        return function;
    }

    std::shared_ptr<const ngraph::Function> reference() const {
        const auto imageParam = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 800, 1216});
        const auto imageGatherIndices = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i32, ngraph::Shape{300, 64});

        std::vector<std::shared_ptr<ngraph::Node>> shapes;
        for (int i = 0; i < 6; i++) {
            const auto gatherDataShape = std::make_shared<ngraph::opset6::ShapeOf>(imageParam, ngraph::element::i32);
            const auto gatherIndicesShape = std::make_shared<ngraph::opset6::ShapeOf>(imageGatherIndices, ngraph::element::i32);

            const auto gatherOutShape = std::make_shared<ngraph::opset6::Concat>(
                ngraph::NodeVector{
                    std::make_shared<ngraph::opset6::Gather>(
                        gatherDataShape,
                        ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{2}, {0, 1}),
                        ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0})),
                    gatherIndicesShape,
                    std::make_shared<ngraph::opset6::Gather>(
                        gatherDataShape,
                        ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {3}),
                        ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0}))},
                0);

            shapes.push_back(
                std::make_shared<ngraph::opset6::Gather>(
                    gatherOutShape,
                    ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {2, 1, 3, 4}),
                    ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0})));
        }

        const auto gatherShape1 = std::make_shared<ngraph::opset6::Gather>(
            shapes[0],
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0}),
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0}));
        const auto gatherShape2 = std::make_shared<ngraph::opset6::Gather>(
            shapes[1],
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{}, {1}),
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0}));
        const auto gatherShape3 = std::make_shared<ngraph::opset6::Gather>(
            shapes[2],
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{}, {2}),
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0}));
        const auto gatherShape4 = std::make_shared<ngraph::opset6::Gather>(
            shapes[3],
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0}),
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0}));
        const auto gatherShape5 = std::make_shared<ngraph::opset6::Gather>(
            shapes[4],
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{}, {1}),
            ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0}));
        const auto gatherShape6 = std::make_shared<ngraph::opset6::Gather>(
            shapes[5],
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

        const auto broadcastParam1 = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i32, ngraph::Shape{300, 1, 1, 64});
        const auto broadcast1 = std::make_shared<ngraph::opset6::Broadcast>(broadcastParam1, select1);

        const auto squeezeAxis = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {0});
        const auto transposePerm = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{4}, {1, 0, 2, 3});

        const auto transposeIndices1 = std::make_shared<ngraph::opset6::Transpose>(
            broadcast1,
            transposePerm);
        const auto unsqueezeIndices1 = std::make_shared<ngraph::opset6::Unsqueeze>(
            transposeIndices1,
            squeezeAxis);

        const auto broadcastParam2 = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::i32, ngraph::Shape{300, 1, 1, 64});
        const auto broadcast2 = std::make_shared<ngraph::opset6::Broadcast>(broadcastParam2, select2);

        const auto transposeIndices2 = std::make_shared<ngraph::opset6::Transpose>(
            broadcast2,
            transposePerm);
        const auto unsqueezeIndices2 = std::make_shared<ngraph::opset6::Unsqueeze>(
            transposeIndices2,
            squeezeAxis);

        const auto expGatherElements1 = std::make_shared<ngraph::vpu::op::ExpGatherElements>(imageParam, unsqueezeIndices1, imageGatherIndices, 3, 2);
        const auto expGatherElements2 = std::make_shared<ngraph::vpu::op::ExpGatherElements>(imageParam, unsqueezeIndices2, imageGatherIndices, 3, 2);

        const auto squeezeData1 = std::make_shared<ngraph::opset6::Squeeze>(
            expGatherElements1,
            squeezeAxis);
        const auto transposeData1 = std::make_shared<ngraph::opset6::Transpose>(
            squeezeData1,
            transposePerm);

        const auto squeezeData2 = std::make_shared<ngraph::opset6::Squeeze>(
            expGatherElements2,
            squeezeAxis);
        const auto transposeData2 = std::make_shared<ngraph::opset6::Transpose>(
            squeezeData2,
            transposePerm);

        return std::make_shared<ngraph::Function>(
            ngraph::OutputVector{transposeData1->output(0), transposeData2->output(0)},
            ngraph::ParameterVector{imageParam, imageGatherIndices, broadcastParam1, broadcastParam2},
            "GatherGatherElements");
    }
};

TEST_F(MergeGatherGatherElements, CompareFunctions) {
}

} // namespace
