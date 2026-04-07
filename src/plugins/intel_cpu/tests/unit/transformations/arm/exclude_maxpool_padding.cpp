// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <limits>
#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/cpu_opset/arm/pass/exclude_maxpool_padding.hpp"

using namespace ov;
using namespace ov::intel_cpu;

namespace {

Output<Node> make_pad(const Output<Node>& input, const Shape& pads_begin, const Shape& pads_end) {
    std::vector<int64_t> pad_begin_values{0, 0};
    pad_begin_values.insert(pad_begin_values.end(), pads_begin.begin(), pads_begin.end());

    std::vector<int64_t> pad_end_values{0, 0};
    pad_end_values.insert(pad_end_values.end(), pads_end.begin(), pads_end.end());

    const auto pad_begin = op::v0::Constant::create(element::i64, Shape{pad_begin_values.size()}, pad_begin_values);
    const auto pad_end = op::v0::Constant::create(element::i64, Shape{pad_end_values.size()}, pad_end_values);
    const auto minus_inf = op::v0::Constant::create(element::f32, Shape{}, {-std::numeric_limits<float>::infinity()});
    const auto pad_value = std::make_shared<op::v1::ConvertLike>(minus_inf, input);

    return std::make_shared<op::v12::Pad>(input, pad_begin, pad_end, pad_value, op::PadMode::CONSTANT);
}

std::shared_ptr<Model> create_v1_model(bool transformed) {
    const auto data = std::make_shared<op::v0::Parameter>(element::i8, Shape{1, 3, 8, 8});
    const Shape pads_begin{1, 2};
    const Shape pads_end{0, 1};

    const Output<Node> input = transformed ? make_pad(data, pads_begin, pads_end) : data;
    const auto max_pool = std::make_shared<op::v1::MaxPool>(input,
                                                            Strides{1, 1},
                                                            transformed ? Shape{0, 0} : pads_begin,
                                                            transformed ? Shape{0, 0} : pads_end,
                                                            Shape{3, 3},
                                                            op::RoundingType::FLOOR,
                                                            transformed ? op::PadType::VALID : op::PadType::EXPLICIT);

    return std::make_shared<Model>(OutputVector{max_pool}, ParameterVector{data});
}

std::shared_ptr<Model> create_v8_model(bool transformed, bool with_indices_result = false) {
    const auto data = std::make_shared<op::v0::Parameter>(element::i8, Shape{1, 3, 8, 8});
    const Shape pads_begin{1, 2};
    const Shape pads_end{0, 1};

    const Output<Node> input = transformed ? make_pad(data, pads_begin, pads_end) : data;
    const auto max_pool = std::make_shared<op::v8::MaxPool>(input,
                                                            Strides{1, 1},
                                                            Strides{1, 1},
                                                            transformed ? Shape{0, 0} : pads_begin,
                                                            transformed ? Shape{0, 0} : pads_end,
                                                            Shape{3, 3},
                                                            op::RoundingType::FLOOR,
                                                            transformed ? op::PadType::VALID : op::PadType::EXPLICIT,
                                                            element::i32,
                                                            0);

    OutputVector outputs{max_pool->output(0)};
    if (with_indices_result) {
        outputs.push_back(max_pool->output(1));
    }

    return std::make_shared<Model>(outputs, ParameterVector{data});
}

std::shared_ptr<Model> create_v14_model(bool transformed) {
    const auto data = std::make_shared<op::v0::Parameter>(element::i8, Shape{1, 3, 8, 8});
    const Shape pads_begin{1, 2};
    const Shape pads_end{0, 1};

    const Output<Node> input = transformed ? make_pad(data, pads_begin, pads_end) : data;
    const auto max_pool = std::make_shared<op::v14::MaxPool>(input,
                                                             Strides{1, 1},
                                                             Strides{1, 1},
                                                             transformed ? Shape{0, 0} : pads_begin,
                                                             transformed ? Shape{0, 0} : pads_end,
                                                             Shape{3, 3},
                                                             op::RoundingType::FLOOR,
                                                             transformed ? op::PadType::VALID : op::PadType::EXPLICIT,
                                                             element::i32,
                                                             0);

    return std::make_shared<Model>(OutputVector{max_pool}, ParameterVector{data});
}

void run_and_compare(const std::shared_ptr<Model>& model, const std::shared_ptr<Model>& model_ref) {
    pass::Manager manager;
    manager.register_pass<ExcludeMaxPoolPadding>();
    manager.run_passes(model);

    const auto comparison = compare_functions(model, model_ref);
    ASSERT_TRUE(comparison.first) << comparison.second;
}

}  // namespace

TEST(TransformationTests, ExcludeMaxPoolPaddingV1) {
    run_and_compare(create_v1_model(false), create_v1_model(true));
}

TEST(TransformationTests, ExcludeMaxPoolPaddingV8) {
    run_and_compare(create_v8_model(false), create_v8_model(true));
}

TEST(TransformationTests, ExcludeMaxPoolPaddingV14) {
    run_and_compare(create_v14_model(false), create_v14_model(true));
}

TEST(TransformationTests, ExcludeMaxPoolPaddingKeepsIndicesPath) {
    run_and_compare(create_v8_model(false, true), create_v8_model(false, true));
}