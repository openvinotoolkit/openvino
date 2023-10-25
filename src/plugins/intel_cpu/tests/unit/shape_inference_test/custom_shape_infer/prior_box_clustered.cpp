// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "custom_shape_infer.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/prior_box_clustered.hpp"
#include "openvino/op/ops.hpp"
namespace ov {
namespace intel_cpu {
namespace unit_test {
namespace cpu_shape_infer {

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

using PriorBoxClusteredV0TestParams = std::tuple<unit_test::ShapeVector,                // Input shapes
                                                 op::v0::PriorBoxClustered::Attributes,
                                                 std::vector<std::vector<int32_t>>,     // layer_data, image_data
                                                 StaticShape                            // Expected shape
                                                 >;

class PriorBoxClusteredV0CpuShapeInferenceTest  : public unit_test::OpCpuShapeInferenceTest<op::v0::PriorBoxClustered>,
                                                  public WithParamInterface<PriorBoxClusteredV0TestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PriorBoxClusteredV0TestParams>& obj) {
        unit_test::ShapeVector tmp_input_shapes;
        op::v0::PriorBoxClustered::Attributes tmp_attrs;
        std::vector<std::vector<int32_t>> tmp_data;
        StaticShape tmp_exp_shape;
        std::tie(tmp_input_shapes, tmp_attrs, tmp_data, tmp_exp_shape) = obj.param;
        std::ostringstream result;

        result << "IS" << ov::test::utils::vec2str(tmp_input_shapes) << "_";
        result << "widths" << ov::test::utils::vec2str(tmp_attrs.widths) << "_";
        result << "heights" << ov::test::utils::vec2str(tmp_attrs.heights) << "_";
        result << "clip(" << unit_test::boolToString(tmp_attrs.clip) << ")_";
        result << "step_widths(" << tmp_attrs.step_widths<< ")_";
        result << "step_heights(" << tmp_attrs.step_heights << ")_";
        result << "offset(" << tmp_attrs.offset << ")_";
        result << "variances" << ov::test::utils::vec2str(tmp_attrs.variances) << "_";
        result << "exp_shape(" << tmp_exp_shape << ")";
        return result.str();
    }

protected:
    void SetUp() override {
        std::tie(input_shapes, attrs, data, exp_shape) = GetParam();
        output_shapes = unit_test::ShapeVector(0);
        output_shapes.push_back(exp_shape);
        ASSERT_LE(input_shapes.size(), 2);
        ASSERT_LE(data.size(), 2);
    }

    op::v0::PriorBoxClustered::Attributes attrs;
    std::vector<std::vector<int32_t>> data;
    unit_test::ShapeVector input_shapes;
};

namespace prior_box_cluster {
const op::v0::PriorBoxClustered::Attributes createAttrs(
    std::vector<float> widths,
    std::vector<float> heights,
    bool clip,
    float step_widths,
    float step_heights,
    float step,
    float offset,
    std::vector<float> variances);

const op::v0::PriorBoxClustered::Attributes createAttrs(
    std::vector<float> widths,
    std::vector<float> heights,
    bool clip,
    float step_widths,
    float step_heights,
    float step,
    float offset,
    std::vector<float> variances) {
    op::v0::PriorBoxClustered::Attributes attrs;
    attrs.widths = widths;
    attrs.heights = heights;
    attrs.clip = clip;
    attrs.step_widths = step_widths;
    attrs.step_heights = step_heights;
    attrs.offset = offset;
    attrs.variances = variances;
    return attrs;
}

const op::v0::PriorBoxClustered::Attributes attrs1 = createAttrs(
    {2.0f, 3.0f} , // widths         Desired widths of prior boxes
    {1.5f, 2.0f},  // heights        Desired heights of prior boxes
    true,          // clip           Clip output to [0,  1]
    0.0f,          // step_widths    Distance between prior box centers
    0.0f,          // step_heights   Distance between prior box centers
    0.0f,          // step           Distance between prior box centers (when step_w = step_h)
    0.0f,          // offset         Box offset relative to top center of image
    {}             // variances      Values to adjust prior boxes with
);

const op::v0::PriorBoxClustered::Attributes attrs2 = createAttrs(
    {86.0f, 13.0f, 57.0f, 39.0f, 68.0f, 34.0f, 142.0f, 50.0f, 23.0f}, // widths         Desired widths of prior boxes
    {44.0f, 10.0f, 30.0f, 19.0f, 94.0f, 32.0f, 61.0f, 53.0f, 17.0f},  // heights        Desired heights of prior boxes
    false,                                                            // clip           Clip output to [0,  1]
    0.0f,                                                             // step_widths    Distance between prior box centers
    0.0f,                                                             // step_heights   Distance between prior box centers
    16.0f,                                                            // step           Distance between prior box centers (when step_w = step_h)
    0.5f,                                                             // offset         Box offset relative to top center of image
    {0.1f, 0.1f, 0.2f, 0.2f}                                          // variances      Values to adjust prior boxes with
);

const op::v0::PriorBoxClustered::Attributes attrs3 = createAttrs(
    {4.0f, 2.0f, 3.2f} , // widths         Desired widths of prior boxes
    {1.0f, 2.0f, 1.1f},  // heights        Desired heights of prior boxes
    true,                // clip           Clip output to [0,  1]
    0.0f,                // step_widths    Distance between prior box centers
    0.0f,                // step_heights   Distance between prior box centers
    0.0f,                // step           Distance between prior box centers (when step_w = step_h)
    0.0f,                // offset         Box offset relative to top center of image
    {}                   // variances      Values to adjust prior boxes with
);

} // namespace prior_box_cluster

TEST_P(PriorBoxClusteredV0CpuShapeInferenceTest , shape_inference_empty_const_map) {
    const auto layer_const = std::make_shared<op::v0::Constant>(element::i32, ov::Shape{2}, data[0]);
    std::shared_ptr<op::v0::PriorBoxClustered> op;
    if (input_shapes.size() == 2) {
        const auto image_const = std::make_shared<op::v0::Constant>(element::i32, ov::Shape{2}, data[1]);
        op = make_op(layer_const, image_const, attrs);
    } else {
        const auto image_param = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
        op = make_op(layer_const, image_param, attrs);
    }
    unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes);
}

TEST_P(PriorBoxClusteredV0CpuShapeInferenceTest , shape_inference_with_const_map) {
    const auto layer_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto image_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto op = make_op(layer_shape, image_shape, attrs);
    std::unordered_map<size_t, ov::Tensor> const_data{{0, {element::i32, ov::Shape{2}, data[0].data()}}};

    if (input_shapes.size() == 2) {
        const_data.insert({1, {element::i32, ov::Shape{2}, data[1].data()}});
    }
    unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes, const_data);
}

INSTANTIATE_TEST_SUITE_P(
    CpuShapeInfer,
    PriorBoxClusteredV0CpuShapeInferenceTest ,
    Values(make_tuple(unit_test::ShapeVector{{2}}, prior_box_cluster::attrs1,
                        std::vector<std::vector<int32_t>>{{2, 5}}, StaticShape({2, 80})),
           make_tuple(unit_test::ShapeVector{{2}, {2}}, prior_box_cluster::attrs1,
                        std::vector<std::vector<int32_t>>{{12, 16}, {50, 50}}, StaticShape({2, 1536})),
           make_tuple(unit_test::ShapeVector{{2}, {2}}, prior_box_cluster::attrs2,
                        std::vector<std::vector<int32_t>>{{10, 19}, {180, 300}}, StaticShape({2, 6840})),
           make_tuple(unit_test::ShapeVector{{2}, {2}}, prior_box_cluster::attrs3,
                        std::vector<std::vector<int32_t>>{{19, 19}, {300, 300}}, StaticShape({2, 4332}))),
    PriorBoxClusteredV0CpuShapeInferenceTest::getTestCaseName);

} // namespace cpu_shape_infer
} // namespace unit_test
} // namespace intel_cpu
} // namespace ov
