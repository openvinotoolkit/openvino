// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
//
//#include <gtest/gtest.h>
//
//#include <string>
//#include <memory>
//
//#include <ngraph/function.hpp>
//#include <ngraph/opsets/opset5.hpp>
//#include <ngraph/pattern/op/wrap_type.hpp>
//#include <ngraph/pass/manager.hpp>
//#include <transformations/common_optimizations/broadcast_elementwise_fusion.hpp>
//#include <transformations/init_node_info.hpp>
//#include <transformations/utils/utils.hpp>
//
//#include "common_test_utils/ngraph_test_utils.hpp"
//
//#define DYN ngraph::Dimension::dynamic()
//
//using namespace testing;
//using namespace ngraph;
//
//using InputShape = PartialShape;
//using TargetShape = Shape;
//
//TEST(TransformationTests, BroadcastElementwiseFusion) {
//    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
//    {
//        auto input_shape = InputShape{DYN, 3, 64, 64, 64};
//        auto target_shape = TargetShape{8, 3, 64, 64, 64};
//        auto input1 =  std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input_shape);
//        auto input2 =  std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, target_shape);
//        auto target_shape_node = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{target_shape.size()}, target_shape);
//        auto broadcast = std::make_shared<ngraph::opset5::Broadcast>(input1, target_shape_node);
//        auto elementwise = std::make_shared<ngraph::opset5::Multiply>(input2, broadcast);
//
//        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{elementwise}, ngraph::ParameterVector{input1, input2});
//
//        ngraph::pass::Manager manager;
//        manager.register_pass<ngraph::pass::InitNodeInfo>();
//        manager.register_pass<ngraph::pass::BroadcastElementwiseFusion>();
//        manager.run_passes(f);
//        ASSERT_NO_THROW(check_rt_info(f));
//    }
//
//    {
//        auto ref_target_shape1 = InputShape{8, 3, 64, 64, 64};
//        auto ref_target_shape2 = InputShape{DYN, 3, 64, 64, 64};
//        auto ref_input1 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ref_target_shape1);
//        auto ref_input2 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ref_target_shape2);
//        auto ref_elementwise = std::make_shared<ngraph::opset5::Multiply>(ref_input1, ref_input2);
//
//        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ref_elementwise}, ngraph::ParameterVector{ref_input1, ref_input2});
//    }
//
//    auto res = compare_functions(f, f_ref);
//    ASSERT_TRUE(res.first) << res.second;
//}
//
//TEST(TransformationTests, BroadcastElementwiseFusionSwitchInput) {
//    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
//    {
//        auto input_shape = InputShape{DYN, 3, 64, 64, 64};
//        auto target_shape = TargetShape{8, 3, 64, 64, 64};
//        auto input1 =  std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input_shape);
//        auto input2 =  std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, target_shape);
//        auto target_shape_node = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{target_shape.size()}, target_shape);
//        auto broadcast = std::make_shared<ngraph::opset5::Broadcast>(input1, target_shape_node);
//        auto elementwise = std::make_shared<ngraph::opset5::Multiply>(broadcast, input2);
//
//        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{elementwise}, ngraph::ParameterVector{input1, input2});
//
//        ngraph::pass::Manager manager;
//        manager.register_pass<ngraph::pass::InitNodeInfo>();
//        manager.register_pass<ngraph::pass::BroadcastElementwiseFusion>();
//        manager.run_passes(f);
//        ASSERT_NO_THROW(check_rt_info(f));
//    }
//
//    {
//        auto ref_target_shape1 = InputShape{DYN, 3, 64, 64, 64};
//        auto ref_target_shape2 = InputShape{8, 3, 64, 64, 64};
//        auto ref_input1 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ref_target_shape1);
//        auto ref_input2 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, ref_target_shape2);
//        auto ref_elementwise = std::make_shared<ngraph::opset5::Multiply>(ref_input1, ref_input2);
//
//        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ref_elementwise}, ngraph::ParameterVector{ref_input1, ref_input2});
//    }
//
//    auto res = compare_functions(f, f_ref);
//    ASSERT_TRUE(res.first) << res.second;
//}


#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <queue>
#include <map>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <transformations/op_conversions/convert_broadcast3.hpp>
#include <transformations/common_optimizations/broadcast_elementwise_fusion.hpp>
#include <ngraph_ops/convolution_ie.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

using InputShape = PartialShape;
using TargetShape = Shape;

void eliminate_broadcast_test(std::shared_ptr<Function> f, std::shared_ptr<Function> f_ref) {
    pass::Manager manager;
    manager.register_pass<ngraph::pass::BroadcastElementwiseFusion>();
    manager.run_passes(f);
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

class EliminateBroadcastTest: public CommonTestUtils::TestsCommon,
                                  public testing::WithParamInterface<std::tuple<InputShape, InputShape, TargetShape>> {
public:
    std::shared_ptr<Function> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& broadcast_input_shape = std::get<1>(GetParam());
        const auto& broadcast_shape = std::get<2>(GetParam());

        f = get_initial_function(input_shape, broadcast_input_shape, broadcast_shape);
        f_ref = get_reference(input_shape, broadcast_shape);
    }

    std::shared_ptr<Function> get_initial_function(const InputShape & input_shape,
                                                   const InputShape & broadcast_input_shape,
                                                   const TargetShape & broadcast_shape) {
        auto input1 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, input_shape);
        auto input2 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, broadcast_input_shape);
        auto input_shape_node = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{broadcast_shape.size()}, broadcast_shape);
        auto broadcast = std::make_shared<ngraph::opset5::Broadcast>(input2, input_shape_node);
        auto elementwise = std::make_shared<ngraph::opset5::Multiply>(input1, broadcast);
        return std::make_shared<ngraph::Function>(ngraph::NodeVector{elementwise}, ngraph::ParameterVector{input1, input2});
    }

    std::shared_ptr<Function> get_reference(const InputShape & input_shape,
                                            const InputShape & broadcast_output_shape) {
        auto ref_input1 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, input_shape);
        auto ref_input2 =  std::make_shared<ngraph::opset5::Parameter>(ngraph::element::f32, broadcast_output_shape);
        auto ref_elementwise = std::make_shared<ngraph::opset5::Multiply>(ref_input1, ref_input2);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{ref_elementwise}, ngraph::ParameterVector{ref_input1, ref_input2});
    }
};

TEST_P(EliminateBroadcastTest, CompareFunctions) {
    eliminate_broadcast_test(f, f_ref);
}

INSTANTIATE_TEST_CASE_P(EliminateBroadcast, EliminateBroadcastTest,
                        testing::Values(std::make_tuple(InputShape{1,2,3}, InputShape{1,2,3}, TargetShape{1,2,3}),
                                        std::make_tuple(InputShape{DYN,2,3}, InputShape{1,2,3}, TargetShape{1,2,3}),
                                        std::make_tuple(InputShape{DYN,DYN,DYN}, InputShape{1,1,1}, TargetShape{1,1,1}),
                                        std::make_tuple(InputShape{1,2,3}, InputShape{2,3}, TargetShape{2,3}),
                                        std::make_tuple(InputShape{1,2,1}, InputShape{1}, TargetShape{1})
                                        //std::make_tuple(InputShape{2,2,4}, TargetShape{2,DYN,4})
                                        ));
