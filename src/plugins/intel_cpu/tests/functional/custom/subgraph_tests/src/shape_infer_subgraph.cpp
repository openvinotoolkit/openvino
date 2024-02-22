// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

class ShapeInferSubgraphTest : virtual public SubgraphBaseTest {
public:
    void run() override {
        ov::element::Type netPrecision = inType = outType = ov::element::f32;
        targetDevice = ov::test::utils::DEVICE_CPU;

        ov::ParameterVector params{
            std::make_shared<ov::op::v0::Parameter>(netPrecision, ov::PartialShape({-1, -1, -1}))};

        auto const_op = [](const std::vector<int>& values) {
            return op::v0::Constant::create(ElementType::i64, {values.size()}, values);
        };

        auto shapeOf = std::make_shared<ov::op::v0::ShapeOf>(params[0]);
        auto gather1 = std::make_shared<ov::op::v8::Gather>(shapeOf, const_op({0}), const_op({0}));
        auto gather2 = std::make_shared<ov::op::v8::Gather>(shapeOf, const_op({1, 2}), const_op({0}));
        auto concat =
            std::make_shared<ov::op::v0::Concat>(ov::NodeVector{gather1, const_op({32}), gather2, const_op({128})}, 0);

        auto gather3 = std::make_shared<ov::op::v8::Gather>(shapeOf, const_op({1}), const_op({0}));
        auto add = std::make_shared<ov::op::v1::Add>(gather1, gather3);
        auto scatter_update =
            std::make_shared<ov::op::v3::ScatterUpdate>(const_op({0, 0}), const_op({1}), add, const_op({0}));

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(concat),
                                 std::make_shared<ov::op::v0::Result>(scatter_update)};
        function = std::make_shared<ov::Model>(results, params, "shape_infer");

        std::vector<ov::Shape> input_shapes = {{4, 2, 3}};
        init_input_shapes(ov::test::static_shapes_to_test_representation(input_shapes));
        ov::test::SubgraphBaseTest::run();
    }
};

namespace {
TEST_F(ShapeInferSubgraphTest, smoke_ShapeInferSubgraphTest_CPU) {
    run();
}
}  // namespace
}  // namespace test
}  // namespace ov
