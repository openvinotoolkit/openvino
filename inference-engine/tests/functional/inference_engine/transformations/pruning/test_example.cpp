// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <sstream>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>

#include <ngraph/pass/manager.hpp>

#include <pruning.hpp>
#include <mask_attribute.hpp>
#include "pruning_test_base.hpp"

using namespace testing;
using namespace ngraph;

using InputShape = ngraph::PartialShape;
using WeightsShape = ngraph::Shape;

namespace {
Output<Node> create_constant_with_zeros(const Shape & shape, const Mask & mask) {
    std::vector<double> values(shape_size(shape), 1);
    for (size_t dim = 0; dim < mask.size(); ++dim) {
        for (const auto & dim_value : mask.at(dim)) {
            Coordinate coord_begin(shape.size(), 0);
            coord_begin[dim] = dim_value;

            Coordinate coord_end(shape);
            coord_end[dim] = dim_value + 1;

            NGRAPH_SUPPRESS_DEPRECATED_START
            CoordinateTransform iter(shape, coord_begin, coord_end);
            for (const Coordinate & coord : iter) {
                values[iter.index(coord)] = 0;
            }
            NGRAPH_SUPPRESS_DEPRECATED_END
        }
    }
    return std::make_shared<opset8::Constant>(element::f32, shape, values);
}
}// namespace

class PruneSingleConvolutionTest: public testing::WithParamInterface<std::tuple<InputShape, WeightsShape>>,
                                  virtual public PruningTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::tuple<InputShape, WeightsShape>>& obj) {
        InputShape inputShapes;
        WeightsShape weightsShape;
        std::tie(inputShapes, weightsShape) = obj.param;
        std::ostringstream result;
        result << "IS=" << CommonTestUtils::partialShape2str({inputShapes}) << "_";
        result << "WS=" << CommonTestUtils::vec2str(weightsShape);
        return result.str();
    }
protected:
    void SetUp() override {
        InputShape inputShapes;
        WeightsShape weightsShape;
        std::tie(inputShapes, weightsShape) = this->GetParam();

        auto param = std::make_shared<opset8::Parameter>(element::f32, inputShapes);
        auto weights = create_constant_with_zeros(weightsShape, {{1, 2, 3}, {}, {}, {}});
        auto conv = std::make_shared<opset8::Convolution>(param, weights, Strides(2, 1),
                                                                          CoordinateDiff(2, 0),
                                                                          CoordinateDiff(2, 0),
                                                                          Strides(2, 1));

        // reference function to get reference accuracy results
        function_ref = std::make_shared<ngraph::Function>(OutputVector{conv}, ParameterVector{param}, "Convolution");

        // function which is processed by Pruning transformation
        function = ov::clone_function(*function_ref);

        pass::Manager m;
        m.register_pass<pass::Pruning>();
        m.run_passes(function);

        // Here we can check the results of Pruning (like number of shrinked elements returned somehow by Pruning)
    }
};

TEST_P(PruneSingleConvolutionTest, AccuracyCheck) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(PruneConvolution, PruneSingleConvolutionTest,
                         ::testing::Combine(
                                 ::testing::Values(PartialShape{1, 6, 16, 16}),
                                 ::testing::ValuesIn({Shape{6, 6, 1, 1}, Shape{12, 6, 1, 2}})),
                         PruneSingleConvolutionTest::getTestCaseName);