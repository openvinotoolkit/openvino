// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "custom_shape_infer.hpp"
#include <memory>
#include "openvino/op/ops.hpp"
namespace ov {
namespace intel_cpu {
namespace unit_test {
namespace cpu_shape_infer {

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

using StridedSliceParams = std::tuple<unit_test::ShapeVector,            // Input shapes
                                      std::vector<std::vector<int32_t>>, // data{begin,end,stride}
                                      std::vector<int64_t>,              // begin_mask
                                      std::vector<int64_t>,              // end_mask
                                      StaticShape                        // Expected shape
                                      >;

class StridedSliceCpuShapeInferenceTest  : public unit_test::OpCpuShapeInferenceTest<op::v1::StridedSlice>,
                                           public WithParamInterface<StridedSliceParams> {
public:
    enum DATA_INDEX {
        BEGIN = 0,
        END = 1,
        STRIDE = 2,
    };
    static std::string getTestCaseName(const testing::TestParamInfo<StridedSliceParams>& obj) {
        unit_test::ShapeVector tmp_input_shapes;
        std::vector<std::vector<int32_t>> tmp_data;
        std::vector<int64_t> tmp_begin_mask;
        std::vector<int64_t> tmp_end_mask;
        StaticShape tmp_exp_shape;
        std::tie(tmp_input_shapes, tmp_data, tmp_begin_mask, tmp_end_mask, tmp_exp_shape) = obj.param;
        std::ostringstream result;
        result << "IS" << CommonTestUtils::vec2str(tmp_input_shapes) << "_";
        result << "begin" << CommonTestUtils::vec2str(tmp_data[BEGIN]) << "_";
        result << "end" << CommonTestUtils::vec2str(tmp_data[END]) << "_";
        result << "stride" << CommonTestUtils::vec2str(tmp_data[STRIDE]) << "_";
        result << "begin_mask" << CommonTestUtils::vec2str(tmp_begin_mask) << "_";
        result << "end_mask" << CommonTestUtils::vec2str(tmp_end_mask) << "_";
        result << "exp_shape(" << tmp_exp_shape << ")";
        return result.str();
    }

protected:
    void SetUp() override {
        std::tie(input_shapes, data, begin_mask, end_mask, exp_shape) = GetParam();
        output_shapes = unit_test::ShapeVector(0);
        output_shapes.push_back(exp_shape);
        ASSERT_EQ(input_shapes.size(), 4);
        arg = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    }
    std::vector<std::vector<int32_t>> data;
    std::vector<int64_t> begin_mask;
    std::vector<int64_t> end_mask;
    std::shared_ptr<op::v0::Parameter> arg;
};

TEST_P(StridedSliceCpuShapeInferenceTest , shape_inference_empty_const_map) {
    const auto begin = op::v0::Constant::create(element::i32, input_shapes[1].get_shape(), data[BEGIN]);
    const auto end = op::v0::Constant::create(element::i32, input_shapes[2].get_shape(), data[END]);
    const auto stride = op::v0::Constant::create(element::i32, input_shapes[3].get_shape(), data[STRIDE]);
    const auto op = make_op(arg, begin, end, stride, begin_mask, end_mask);
    // implementation depends on some output information of the op
    op->set_output_type(0, element::i32, {-1, -1, -1});
    unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes);
}

TEST_P(StridedSliceCpuShapeInferenceTest , shape_inference_in_const_map) {
    const auto begin = std::make_shared<op::v0::Parameter>(element::i32, input_shapes[1].get_shape());
    const auto end = std::make_shared<op::v0::Parameter>(element::i32, input_shapes[2].get_shape());
    const auto stride = std::make_shared<op::v0::Parameter>(element::i32, input_shapes[3].get_shape());
    const auto op = make_op(arg, begin, end, stride, begin_mask, end_mask);

    const auto begin_const = std::make_shared<op::v0::Constant>(element::i32, input_shapes[1].get_shape(), data[BEGIN]);
    const auto end_const = std::make_shared<op::v0::Constant>(element::i32, input_shapes[2].get_shape(), data[END]);
    const auto stride_const = std::make_shared<op::v0::Constant>(element::i32, input_shapes[3].get_shape(), data[STRIDE]);
    const auto begin_tensor = std::make_shared<ov::HostTensor>(begin_const);
    const auto end_tensor = std::make_shared<ov::HostTensor>(end_const);
    const auto stride_tensor = std::make_shared<ov::HostTensor>(stride_const);
    const std::map<size_t, ov::HostTensorPtr>& constant_data = {{1, begin_tensor},
                                                                                           {2, end_tensor},
                                                                                           {3, stride_tensor}};
    // implementation depends on some output information of the op
    op->set_output_type(0, element::i32, {-1, -1, -1});
    unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes, constant_data);
}

INSTANTIATE_TEST_SUITE_P(
    CpuShapeInfer,
    StridedSliceCpuShapeInferenceTest,
    Values(make_tuple(unit_test::ShapeVector{{3, 4, 5}, {3}, {3}, {3}}, std::vector<std::vector<int32_t>>{{100}, {-100}, {-1}},
                      std::vector<int64_t>(4, 0), std::vector<int64_t>(4, 0), StaticShape({3, 4, 5})),
           make_tuple(unit_test::ShapeVector{{3, 2, 3}, {3}, {3}, {3}}, std::vector<std::vector<int32_t>>{{1, 0, 0}, {2, 1, 3}, {1, 1, 1}},
                      std::vector<int64_t>(4, 0), std::vector<int64_t>(4, 0), StaticShape({1, 1, 3})),
           make_tuple(unit_test::ShapeVector{{3, 2, 3}, {3}, {3}, {3}}, std::vector<std::vector<int32_t>>{{1, 0, 0}, {2, 2, 3}, {1, 1, 1}},
                      std::vector<int64_t>(4, 0), std::vector<int64_t>(4, 0), StaticShape({1, 2, 3})),
           make_tuple(unit_test::ShapeVector{{3, 2, 3}, {3}, {3}, {3}}, std::vector<std::vector<int32_t>>{{2, 0, 0}, {3, 2, 3}, {1, 1, 2}},
                      std::vector<int64_t>(4, 0), std::vector<int64_t>(4, 0), StaticShape({1, 2, 2})),
           make_tuple(unit_test::ShapeVector{{3, 2, 3}, {3}, {3}, {3}}, std::vector<std::vector<int32_t>>{{1, 0, 0}, {0, 0, 0}, {1, 1, 1}},
                      std::vector<int64_t>{0, 1, 1}, std::vector<int64_t>(3, 1), StaticShape({2, 2, 3})),
           make_tuple(unit_test::ShapeVector{{3, 2, 3}, {3}, {3}, {3}}, std::vector<std::vector<int32_t>>{{0, 1, 0}, {2, 0, 0}, {1, 1, 2}},
                      std::vector<int64_t>{1, 0, 1}, std::vector<int64_t>{0, 1, 1}, StaticShape({2, 1, 2}))),
           // TODO  108946, can't pass;
           // make_tuple(unit_test::ShapeVector{{3, 2, 3}, {3}, {3}, {3}}, std::vector<std::vector<int32_t>>{{0, 0, 0}, {1, 0, 0}, {1, 1, -1}},
           //            std::vector<int64_t>{0, 1, 1}, std::vector<int64_t>{0, 1, 1}, StaticShape({1, 1, 3}))),
    StridedSliceCpuShapeInferenceTest::getTestCaseName);

TEST(CpuShapeInfer, StridedSliceDefault_stride) {
    GTEST_SKIP() << "Skipping test, please check CVS-108946";
    const auto mask = std::vector<int64_t>{0, 1, 0};

    const auto data = std::make_shared<op::v0::Parameter>(element::f32, ov::PartialShape::dynamic());
    // only supprot i32
    const auto begin = op::v0::Constant::create(element::i32, ov::Shape{3}, {0, 0, 0});
    const auto end = op::v0::Constant::create(element::i32, ov::Shape{3}, {1, 0, 2});
    const auto op = std::make_shared<op::v1::StridedSlice>(data, begin, end, mask, mask);

    std::vector<StaticShape> static_input_shapes = {{3, 2, 3}, {3}, {3}};
    std::vector<StaticShape> static_output_shapes = {StaticShape{1, 2, 2}};
    //  implementation depends on some output information of the op
    op->set_output_type(0, element::i32, {-1, -1, -1});
    // TODO 108946,there is some issue in implementation, this test case can't pass
    unit_test::cpu_test_shape_infer(op.get(), static_input_shapes, static_output_shapes);
}
} // namespace cpu_shape_infer
} // namespace unit_test
} // namespace intel_cpu
} // namespace ov

