// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "custom_shape_infer.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/prior_box.hpp"
#include "openvino/op/parameter.hpp"

namespace ov {
namespace intel_cpu {
namespace unit_test {
namespace cpu_shape_infer {

using namespace ov;
using namespace ov::intel_cpu;
using namespace testing;

// TODO should support v8::PriorBox

using PriorBoxV0TestParams = std::tuple<unit_test::ShapeVector,            // Input shapes
                                        op::v0::PriorBox::Attributes,
                                        std::vector<std::vector<int32_t>>, // layer_data, image_data
                                        StaticShape                        // Expected shape
                                        >;

class PriorBoxV0CpuShapeInferenceTest  : public unit_test::OpCpuShapeInferenceTest<ov::op::v0::PriorBox>,
                                      public WithParamInterface<PriorBoxV0TestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PriorBoxV0TestParams>& obj) {
        const auto& [tmp_input_shapes, tmp_attrs, tmp_data, tmp_exp_shape] = obj.param;
        std::ostringstream result;
        result << "IS" << ov::test::utils::vec2str(tmp_input_shapes) << "_";
        result << "min_size" << ov::test::utils::vec2str(tmp_attrs.min_size) << "_";
        result << "max_size" << ov::test::utils::vec2str(tmp_attrs.max_size) << "_";
        result << "density" << ov::test::utils::vec2str(tmp_attrs.density) << "_";
        result << "fixed_ratio" << ov::test::utils::vec2str(tmp_attrs.fixed_ratio) << "_";
        result << "fixed_size" << ov::test::utils::vec2str(tmp_attrs.fixed_size) << "_";
        result << "clip(" << unit_test::boolToString(tmp_attrs.clip) << ")_";
        result << "flip(" << unit_test::boolToString(tmp_attrs.flip) << ")_";
        result << "step(" << tmp_attrs.step << ")_";
        result << "offset(" << tmp_attrs.offset << ")_";
        result << "variance" << ov::test::utils::vec2str(tmp_attrs.variance) << "_";
        result << "scale_all_sizes(" << unit_test::boolToString(tmp_attrs.scale_all_sizes) << ")_";
        result << "exp_shape(" << tmp_exp_shape << ")";
        return result.str();
    }

protected:
    void SetUp() override {
        std::tie(input_shapes, attrs, data, exp_shape) = GetParam();
        output_shapes = unit_test::ShapeVector(0);
        output_shapes.push_back(exp_shape);
        ASSERT_EQ(input_shapes.size(), 2);
        ASSERT_EQ(data.size(), 2);
    }

    op::v0::PriorBox::Attributes attrs;
    std::vector<std::vector<int32_t>> data;
};

namespace prior_box {
const op::v0::PriorBox::Attributes createAttrs(
    std::vector<float> min_size,
    std::vector<float> max_size,
    std::vector<float> aspect_ratio,
    std::vector<float> density,
    std::vector<float> fixed_ratio,
    std::vector<float> fixed_size,
    bool clip,
    bool flip,
    float step,
    float offset,
    std::vector<float> variance,
    bool scale_all_sizes);
const op::v0::PriorBox::Attributes createAttrs(
    std::vector<float> min_size,
    std::vector<float> max_size,
    std::vector<float> aspect_ratio,
    std::vector<float> density,
    std::vector<float> fixed_ratio,
    std::vector<float> fixed_size,
    bool clip,
    bool flip,
    float step,
    float offset,
    std::vector<float> variance,
    bool scale_all_sizes) {
    op::v0::PriorBox::Attributes attrs;
    attrs.min_size = min_size;
    attrs.max_size = max_size;
    attrs.aspect_ratio = aspect_ratio;
    attrs.density = density;
    attrs.fixed_ratio = fixed_ratio;
    attrs.fixed_size = fixed_size;
    attrs.clip = clip;
    attrs.flip = flip;
    attrs.step = step;
    attrs.offset = offset;
    attrs.variance = variance;
    attrs.scale_all_sizes = scale_all_sizes;
    return attrs;
}
const op::v0::PriorBox::Attributes attrs1 = createAttrs(
    {16.0f},                  // min_size         Desired min_size of prior boxes
    {38.46f},                 // max_size         Desired max_size of prior boxes
    {2.0f},                   // aspect_ratio     Aspect ratios of prior boxes
    {},                       // density
    {},                       // fixed_ratio
    {},                       // fixed_size
    false,                    // clip             Clip output to [0,  1]
    true,                     // flip             Flip aspect ratios
    16.0f,                    // step             Distance between prior box centers
    0.5f,                     // offset           Box offset relative to top center of image
    {0.1f, 0.1f, 0.2f, 0.2f}, // variance         Values to adjust prior boxes with
    true                      // scale_all_sizes  Scale all sizes
);

const op::v0::PriorBox::Attributes attrs2 = createAttrs(
    {2.0f, 3.0f},       // min_size         Desired min_size of prior boxes
    {},                 // max_size         Desired max_size of prior boxes
    {1.5f, 2.0f, 2.5f}, // aspect_ratio     Aspect ratios of prior boxes
    {},                 // density
    {},                 // fixed_ratio
    {},                 // fixed_size
    false,              // clip             Clip output to [0,  1]
    false,              // flip             Flip aspect ratios
    0.0f,               // step             Distance between prior box centers
    0.0f,               // offset           Box offset relative to top center of image
    {},                 // variance         Values to adjust prior boxes with
    false               // scale_all_sizes  Scale all sizes
);

const op::v0::PriorBox::Attributes attrs3 = createAttrs(
    {2.0f, 3.0f},       // min_size         Desired min_size of prior boxes
    {},                 // max_size         Desired max_size of prior boxes
    {1.5f, 2.0f, 2.5f}, // aspect_ratio     Aspect ratios of prior boxes
    {},                 // density
    {},                 // fixed_ratio
    {},                 // fixed_size
    false,              // clip             Clip output to [0,  1]
    true,               // flip             Flip aspect ratios
    0.0f,               // step             Distance between prior box centers
    0.0f,               // offset           Box offset relative to top center of image
    {},                 // variance         Values to adjust prior boxes with
    false               // scale_all_sizes  Scale all sizes
);

const op::v0::PriorBox::Attributes attrs4 = createAttrs(
    {256.0f}, // min_size         Desired min_size of prior boxes
    {315.0f}, // max_size         Desired max_size of prior boxes
    {2.0f},   // aspect_ratio     Aspect ratios of prior boxes
    {},       // density
    {},       // fixed_ratio
    {},       // fixed_size
    false,    // clip             Clip output to [0,  1]
    true,     // flip             Flip aspect ratios
    0.0f,     // step             Distance between prior box centers
    0.0f,     // offset           Box offset relative to top center of image
    {},       // variance         Values to adjust prior boxes with
    true      // scale_all_sizes  Scale all sizes
);

} // namespace prior_box

TEST_P(PriorBoxV0CpuShapeInferenceTest , shape_inference_empty_const_map) {
    const auto layer_const = std::make_shared<op::v0::Constant>(element::i32, ov::Shape{2}, data[0]);
    const auto image_const = std::make_shared<op::v0::Constant>(element::i32, ov::Shape{2}, data[1]);
    auto op = make_op(layer_const, image_const, attrs);
    unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes);
}

TEST_P(PriorBoxV0CpuShapeInferenceTest , shape_inference_with_const_map) {
    const auto layer_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto image_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto op = make_op(layer_shape, image_shape, attrs);

    const std::unordered_map<size_t, ov::Tensor> const_data{{0, {element::i32, ov::Shape{2}, data[0].data()}},
                                                            {1, {element::i32, ov::Shape{2}, data[1].data()}}};

    unit_test::cpu_test_shape_infer(op.get(), input_shapes, output_shapes, const_data);
}

INSTANTIATE_TEST_SUITE_P(
    CpuShapeInfer,
    PriorBoxV0CpuShapeInferenceTest ,
    Values(make_tuple(unit_test::ShapeVector{{2}, {2}}, prior_box::attrs1,
                        std::vector<std::vector<int32_t>>{{24, 42}, {384, 672}}, StaticShape({2, 16128})),
           make_tuple(unit_test::ShapeVector{{2}, {2}}, prior_box::attrs2,
                        std::vector<std::vector<int32_t>>{{32, 32}, {384, 672}}, StaticShape({2, 20480})),
           make_tuple(unit_test::ShapeVector{{2}, {2}}, prior_box::attrs3,
                        std::vector<std::vector<int32_t>>{{32, 32}, {300, 300}}, StaticShape({2, 32768})),
           make_tuple(unit_test::ShapeVector{{2}, {2}}, prior_box::attrs4,
                        std::vector<std::vector<int32_t>>{{1, 1}, {300, 300}}, StaticShape({2, 16}))),
    PriorBoxV0CpuShapeInferenceTest::getTestCaseName);

} // namespace cpu_shape_infer
} // namespace unit_test
} // namespace intel_cpu
} // namespace ov
