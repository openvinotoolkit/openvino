// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <limits>
#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/cpu_opset/arm/pass/exclude_maxpool_padding.hpp"

using namespace ov;
using namespace ov::intel_cpu;

namespace {

std::shared_ptr<op::v0::Constant> make_pad_value(const element::Type& type, bool use_zero_pad_value = false) {
    if (use_zero_pad_value) {
        return op::v0::Constant::create(type, Shape{}, {0.f});
    }

    if (type == element::f16) {
        return op::v0::Constant::create(type, Shape{}, {std::numeric_limits<ov::float16>::lowest()});
    }
    if (type == element::bf16) {
        return op::v0::Constant::create(type, Shape{}, {std::numeric_limits<ov::bfloat16>::lowest()});
    }
    if (type == element::f32) {
        return op::v0::Constant::create(type, Shape{}, {std::numeric_limits<float>::lowest()});
    }
    if (type == element::f64) {
        return op::v0::Constant::create(type, Shape{}, {std::numeric_limits<double>::lowest()});
    }

    OPENVINO_THROW("Unsupported precision for ExcludeMaxPoolPadding test pad value: ", type);
}

Output<Node> make_pad(const Output<Node>& input,
                      const Shape& pads_begin,
                      const Shape& pads_end,
                      bool use_zero_pad_value = false) {
    const auto pad_value = make_pad_value(input.get_element_type(), use_zero_pad_value);
    std::vector<int64_t> pad_begin_values{0, 0};
    pad_begin_values.insert(pad_begin_values.end(), pads_begin.begin(), pads_begin.end());

    std::vector<int64_t> pad_end_values{0, 0};
    pad_end_values.insert(pad_end_values.end(), pads_end.begin(), pads_end.end());

    Output<Node> padded = input;
    for (size_t dim = 0; dim < pad_begin_values.size(); ++dim) {
        if (pad_begin_values[dim] == 0 && pad_end_values[dim] == 0) {
            continue;
        }

        std::vector<int64_t> dim_pad_begin_values(pad_begin_values.size(), 0);
        std::vector<int64_t> dim_pad_end_values(pad_end_values.size(), 0);
        dim_pad_begin_values[dim] = pad_begin_values[dim];
        dim_pad_end_values[dim] = pad_end_values[dim];

        const auto pad_begin = op::v0::Constant::create(element::i64,
                                                        Shape{dim_pad_begin_values.size()},
                                                        dim_pad_begin_values);
        const auto pad_end = op::v0::Constant::create(element::i64, Shape{dim_pad_end_values.size()}, dim_pad_end_values);
        padded = std::make_shared<op::v12::Pad>(padded, pad_begin, pad_end, pad_value, op::PadMode::CONSTANT);
    }

    return padded;
}

Output<Node> maybe_make_fake_quantize(const Output<Node>& input,
                                      bool with_fake_quantize,
                                      float output_low = -12.8f,
                                      float output_high = 12.7f) {
    if (!with_fake_quantize) {
        return input;
    }

    return ov::test::utils::make_fake_quantize(input,
                                               element::f32,
                                               256,
                                               Shape{},
                                               {output_low},
                                               {output_high},
                                               {output_low},
                                               {output_high});
}

std::shared_ptr<Model> create_v1_model(bool transformed,
                                       bool with_fake_quantize = false,
                                       const element::Type& precision = element::f16,
                                       bool use_zero_pad_value = false,
                                       float fq_output_low = -12.8f,
                                       float fq_output_high = 12.7f) {
    const auto data = std::make_shared<op::v0::Parameter>(precision, Shape{1, 3, 8, 8});
    const Shape pads_begin{1, 2};
    const Shape pads_end{0, 1};

    const auto pooled_input = maybe_make_fake_quantize(data, with_fake_quantize, fq_output_low, fq_output_high);
    const Output<Node> input = transformed ? make_pad(pooled_input, pads_begin, pads_end, use_zero_pad_value) : pooled_input;
    const auto max_pool = std::make_shared<op::v1::MaxPool>(input,
                                                            Strides{1, 1},
                                                            transformed ? Shape{0, 0} : pads_begin,
                                                            transformed ? Shape{0, 0} : pads_end,
                                                            Shape{3, 3},
                                                            op::RoundingType::FLOOR,
                                                            transformed ? op::PadType::VALID : op::PadType::EXPLICIT);

    return std::make_shared<Model>(OutputVector{max_pool}, ParameterVector{data});
}

std::shared_ptr<Model> create_v8_model(bool transformed,
                                       bool with_indices_result = false,
                                       bool with_fake_quantize = false,
                                       const element::Type& precision = element::f16,
                                       bool use_zero_pad_value = false,
                                       float fq_output_low = -12.8f,
                                       float fq_output_high = 12.7f) {
    const auto data = std::make_shared<op::v0::Parameter>(precision, Shape{1, 3, 8, 8});
    const Shape pads_begin{1, 2};
    const Shape pads_end{0, 1};

    const auto pooled_input = maybe_make_fake_quantize(data, with_fake_quantize, fq_output_low, fq_output_high);
    const Output<Node> input = transformed ? make_pad(pooled_input, pads_begin, pads_end, use_zero_pad_value) : pooled_input;
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

std::shared_ptr<Model> create_v14_model(bool transformed,
                                        bool with_fake_quantize = false,
                                        const element::Type& precision = element::f16,
                                        bool use_zero_pad_value = false,
                                        float fq_output_low = -12.8f,
                                        float fq_output_high = 12.7f) {
    const auto data = std::make_shared<op::v0::Parameter>(precision, Shape{1, 3, 8, 8});
    const Shape pads_begin{1, 2};
    const Shape pads_end{0, 1};

    const auto pooled_input = maybe_make_fake_quantize(data, with_fake_quantize, fq_output_low, fq_output_high);
    const Output<Node> input = transformed ? make_pad(pooled_input, pads_begin, pads_end, use_zero_pad_value) : pooled_input;
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
    manager.get_pass_config()->set_callback<ExcludeMaxPoolPadding>([](const std::shared_ptr<const ov::Node>& node) {
        const auto max_pool = ov::as_type_ptr<const ov::op::util::MaxPoolBase>(node);
        return !max_pool ||
               ov::as_type_ptr<const ov::op::v0::FakeQuantize>(max_pool->get_input_node_shared_ptr(0)) == nullptr;
    });
    manager.run_passes(model);

    const auto comparison = compare_functions(model, model_ref);
    ASSERT_TRUE(comparison.first) << comparison.second;
}

}  // namespace

TEST(TransformationTests, ExcludeMaxPoolPaddingV1) {
    run_and_compare(create_v1_model(false, true), create_v1_model(true, true));
}

TEST(TransformationTests, ExcludeMaxPoolPaddingV8) {
    run_and_compare(create_v8_model(false, false, true), create_v8_model(true, false, true));
}

TEST(TransformationTests, ExcludeMaxPoolPaddingV14) {
    run_and_compare(create_v14_model(false, true), create_v14_model(true, true));
}

TEST(TransformationTests, ExcludeMaxPoolPaddingUsesZeroPadForNonNegativeFakeQuantizeOutput) {
    run_and_compare(create_v8_model(false, false, true, element::f16, false, 0.f, 25.5f),
                    create_v8_model(true, false, true, element::f16, true, 0.f, 25.5f));
}

TEST(TransformationTests, ExcludeMaxPoolPaddingKeepsIndicesPath) {
    run_and_compare(create_v8_model(false, true), create_v8_model(false, true));
}

TEST(TransformationTests, ExcludeMaxPoolPaddingKeepsNonQuantizedV1) {
    run_and_compare(create_v1_model(false, false), create_v1_model(false, false));
}

TEST(TransformationTests, ExcludeMaxPoolPaddingKeepsNonQuantizedV8) {
    run_and_compare(create_v8_model(false, false, false), create_v8_model(false, false, false));
}

TEST(TransformationTests, ExcludeMaxPoolPaddingKeepsNonQuantizedV14) {
    run_and_compare(create_v14_model(false, false), create_v14_model(false, false));
}