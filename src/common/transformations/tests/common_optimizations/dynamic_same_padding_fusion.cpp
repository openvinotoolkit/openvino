// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "transformations/common_optimizations/dynamic_same_padding_fusion.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset13.hpp"
#include "transformations/common_optimizations/moc_transformations.hpp"

namespace {
using namespace ov;
using namespace ov::opset13;

struct Config {
    PartialShape shape{1, 3, -1, -1};
    Shape kernel{3, 3};
    Strides strides{2, 2};
    Strides dilations{1, 1};
    element::Type math_type = element::f32;
    element::Type shape_type = element::i64;
    bool grouped = false;
    bool pad_v1 = false;
    bool expanded = false;
    bool decompose_subtract = false;
    bool reciprocal = true;
    bool reciprocal_stride = false;
    bool timm_layout = true;
    bool wrong_axis = false;
    bool wrong_source = false;
    bool same_lower = false;
    bool shared_conv = false;
    bool different_consumer_stride = false;
    bool output_pad = false;
    bool output_shape = false;
    bool implicit_pad_value = false;
    double pad_value = 0;
    double batch_pad = 0;
    double kernel_adjustment = 0;
    double split_divisor = 2;
    op::PadMode pad_mode = op::PadMode::CONSTANT;
    op::PadType auto_pad = op::PadType::EXPLICIT;
    CoordinateDiff conv_pads{0, 0};
};

Output<Node> scalar_constant(const Config& c, double value) {
    return Constant::create(c.math_type, Shape{}, {value});
}

Output<Node> subtract(const Output<Node>& a, const Output<Node>& b, const Config& c) {
    if (c.decompose_subtract)
        return std::make_shared<Add>(a, std::make_shared<Multiply>(b, scalar_constant(c, -1)));
    return std::make_shared<Subtract>(a, b);
}

std::shared_ptr<op::util::PadBase> make_pad(const Output<Node>& data, const Output<Node>& shape, const Config& c) {
    const auto axis_zero = Constant::create(element::i32, Shape{}, {0});
    OutputVector begins, ends;
    for (size_t i = 0; i < c.kernel.size(); ++i) {
        const auto axis = c.wrong_axis ? 2 + (i + 1) % c.kernel.size() : 2 + i;
        const auto size = std::make_shared<Convert>(
            std::make_shared<Gather>(shape, Constant::create(element::i64, Shape{}, {axis}), axis_zero),
            c.math_type);
        const auto stride = scalar_constant(c, c.strides[i]);
        Output<Node> divided;
        if (c.reciprocal_stride)
            divided = std::make_shared<Multiply>(size, scalar_constant(c, 1.0 / c.strides[i]));
        else
            divided = std::make_shared<Divide>(size, stride);
        const auto ceil = std::make_shared<Ceiling>(divided);
        const auto effective = (c.kernel[i] - 1) * c.dilations[i] + 1 + c.kernel_adjustment;
        Output<Node> extent;
        if (c.expanded) {
            extent = std::make_shared<Add>(std::make_shared<Multiply>(subtract(ceil, scalar_constant(c, 1), c), stride),
                                           scalar_constant(c, effective));
        } else {
            extent = std::make_shared<Add>(std::make_shared<Multiply>(stride, ceil),
                                           scalar_constant(c, effective - c.strides[i]));
        }
        const auto total = std::make_shared<Maximum>(scalar_constant(c, 0), subtract(extent, size, c));
        Output<Node> half;
        if (c.reciprocal)
            half = std::make_shared<Multiply>(scalar_constant(c, 1 / c.split_divisor), total);
        else
            half = std::make_shared<Divide>(total, scalar_constant(c, c.split_divisor));
        half = std::make_shared<Floor>(half);
        Output<Node> before = std::make_shared<Convert>(half, c.shape_type);
        Output<Node> after = std::make_shared<Convert>(subtract(total, half, c), c.shape_type);
        if (c.same_lower)
            std::swap(before, after);
        begins.push_back(std::make_shared<Unsqueeze>(before, axis_zero));
        ends.push_back(std::make_shared<Unsqueeze>(after, axis_zero));
    }
    Output<Node> begin, end;
    if (c.timm_layout) {
        // F.pad uses [W_begin, W_end, H_begin, H_end, ...]. Its frontend
        // translation splits the pairs and reverses the spatial axes.
        OutputVector pairs;
        std::vector<int64_t> reverse;
        for (size_t i = begins.size(); i-- > 0;) {
            pairs.push_back(begins[i]);
            pairs.push_back(ends[i]);
            reverse.push_back(i);
        }
        const auto packed = std::make_shared<Concat>(pairs, 0);
        const auto matrix = std::make_shared<Reshape>(packed, Constant::create(element::i32, Shape{2}, {-1, 2}), false);
        const auto axis_one = Constant::create(element::i32, Shape{}, {1});
        const auto split = std::make_shared<Split>(matrix, axis_one, 2);
        const auto indices = Constant::create(element::i64, Shape{reverse.size()}, reverse);
        begin = std::make_shared<Gather>(std::make_shared<Squeeze>(split->output(0), axis_one), indices, axis_zero);
        end = std::make_shared<Gather>(std::make_shared<Squeeze>(split->output(1), axis_one), indices, axis_zero);
    } else {
        begin = std::make_shared<Concat>(begins, 0);
        end = std::make_shared<Concat>(ends, 0);
    }
    const auto zeros = Constant::create(c.shape_type, Shape{2}, {c.batch_pad, 0.0});
    begin = std::make_shared<Concat>(OutputVector{zeros, begin}, 0);
    end = std::make_shared<Concat>(OutputVector{zeros, end}, 0);
    const auto value = Constant::create(element::f32, Shape{}, {c.pad_value});
    if (c.pad_v1) {
        if (c.implicit_pad_value)
            return std::make_shared<op::v1::Pad>(data, begin, end, c.pad_mode);
        return std::make_shared<op::v1::Pad>(data, begin, end, value, c.pad_mode);
    }
    if (c.implicit_pad_value)
        return std::make_shared<Pad>(data, begin, end, c.pad_mode);
    return std::make_shared<Pad>(data, begin, end, value, c.pad_mode);
}

std::shared_ptr<Node> make_conv(const Output<Node>& data, const Config& c, bool same, const std::string& name) {
    Shape weight_shape = c.grouped ? Shape{3, 1, 1} : Shape{4, 3};
    weight_shape.insert(weight_shape.end(), c.kernel.begin(), c.kernel.end());
    const auto weights = Constant::create(element::f32, weight_shape, {0.25f});
    weights->output(0).get_tensor().set_names({name + "_weights"});
    const auto auto_pad = same ? op::PadType::SAME_UPPER : c.auto_pad;
    std::shared_ptr<Node> conv;
    if (c.grouped)
        conv = std::make_shared<GroupConvolution>(data,
                                                  weights,
                                                  c.strides,
                                                  c.conv_pads,
                                                  c.conv_pads,
                                                  c.dilations,
                                                  auto_pad);
    else
        conv = std::make_shared<Convolution>(data, weights, c.strides, c.conv_pads, c.conv_pads, c.dilations, auto_pad);
    conv->set_friendly_name(name);
    conv->output(0).get_tensor().set_names({name + "_output"});
    return conv;
}

Output<Node> spatial_shape(const Output<Node>& shape) {
    return std::make_shared<Gather>(shape,
                                    Constant::create(element::i64, Shape{2}, {2, 3}),
                                    Constant::create(element::i64, Shape{}, {0}));
}

std::shared_ptr<Model> get_model(const Config& c) {
    const auto data = std::make_shared<Parameter>(element::f32, c.shape);
    data->output(0).get_tensor().set_names({"input"});
    ParameterVector parameters{data};
    Output<Node> shape_source = data;
    if (c.wrong_source) {
        const auto other = std::make_shared<Parameter>(element::f32, c.shape);
        parameters.push_back(other);
        shape_source = other;
    }
    const auto shape = std::make_shared<ShapeOf>(shape_source, c.shape_type);
    const auto pad = make_pad(data, shape, c);
    OutputVector outputs{make_conv(pad, c, false, "conv")};
    if (c.shared_conv) {
        auto other = c;
        if (c.different_consumer_stride)
            other.strides[0] = 1;
        outputs.push_back(make_conv(pad, other, false, "other_conv"));
    }
    if (c.output_pad)
        outputs.push_back(pad);
    if (c.output_shape)
        outputs.push_back(spatial_shape(shape));
    return std::make_shared<Model>(outputs, parameters);
}

std::shared_ptr<Model> get_model_ref(const Config& c) {
    const auto data = std::make_shared<Parameter>(element::f32, c.shape);
    data->output(0).get_tensor().set_names({"input"});
    const auto shape = std::make_shared<ShapeOf>(data, c.shape_type);
    OutputVector outputs{make_conv(data, c, true, "conv")};
    std::shared_ptr<op::util::PadBase> pad;
    if (c.output_pad || c.different_consumer_stride)
        pad = make_pad(data, shape, c);
    if (c.shared_conv) {
        auto other = c;
        if (c.different_consumer_stride) {
            other.strides[0] = 1;
            outputs.push_back(make_conv(pad, other, false, "other_conv"));
        } else {
            outputs.push_back(make_conv(data, other, true, "other_conv"));
        }
    }
    if (c.output_pad)
        outputs.push_back(pad);
    if (c.output_shape)
        outputs.push_back(spatial_shape(shape));
    return std::make_shared<Model>(outputs, ParameterVector{data});
}

class DynamicSamePaddingFusionTests : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        comparator.enable(FunctionsComparator::CONST_VALUES);
        manager.register_pass<pass::DynamicSamePaddingFusion>();
    }
};

using TestParams = std::tuple<bool, bool, bool, size_t>;
class DynamicSamePaddingFusionParameterized : public DynamicSamePaddingFusionTests,
                                              public testing::WithParamInterface<TestParams> {
public:
    static std::string get_test_name(const testing::TestParamInfo<TestParams>& info) {
        const auto& [grouped, pad_v1, expanded, spatial_rank] = info.param;
        return std::string(grouped ? "GroupConv" : "Conv") + (pad_v1 ? "_Pad1" : "_Pad12") +
               (expanded ? "_Expanded" : "_Folded") + "_" + std::to_string(spatial_rank) + "D";
    }
};

TEST_P(DynamicSamePaddingFusionParameterized, SpatialPadding) {
    const auto& [grouped, pad_v1, expanded, spatial_rank] = GetParam();
    Config c;
    c.grouped = grouped;
    c.pad_v1 = pad_v1;
    c.expanded = expanded;
    c.reciprocal = !expanded;
    c.reciprocal_stride = !expanded;
    c.shape = PartialShape::dynamic(spatial_rank + 2);
    c.shape[1] = 3;
    c.kernel.assign(spatial_rank, 3);
    c.strides.assign(spatial_rank, 2);
    c.dilations.assign(spatial_rank, 1);
    c.conv_pads.assign(spatial_rank, 0);
    model = get_model(c);
    model_ref = get_model_ref(c);
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         DynamicSamePaddingFusionParameterized,
                         testing::Combine(testing::Bool(), testing::Bool(), testing::Bool(), testing::Values(1, 2, 3)),
                         DynamicSamePaddingFusionParameterized::get_test_name);

TEST_F(DynamicSamePaddingFusionTests, DecomposedSubtractAndDirectPaddingVectors) {
    Config c;
    c.decompose_subtract = true;
    c.timm_layout = false;
    c.shape_type = element::i32;
    c.math_type = element::f64;
    c.kernel = {2, 5};
    c.strides = {3, 2};
    c.dilations = {2, 3};
    model = get_model(c);
    model_ref = get_model_ref(c);
}

TEST_F(DynamicSamePaddingFusionTests, SharedKeyValueConvolutions) {
    Config c;
    c.grouped = true;
    c.shared_conv = true;
    c.output_shape = true;
    comparator.enable(FunctionsComparator::CONSUMERS_COUNT);
    model = get_model(c);
    model_ref = get_model_ref(c);
}

TEST_F(DynamicSamePaddingFusionTests, PreservePadForIncompatibleConsumer) {
    Config c;
    c.shared_conv = true;
    c.different_consumer_stride = true;
    c.output_pad = true;
    model = get_model(c);
    model_ref = get_model_ref(c);
}

TEST_F(DynamicSamePaddingFusionTests, ImplicitZeroPaddingValue) {
    Config c;
    c.implicit_pad_value = true;
    model = get_model(c);
    model_ref = get_model_ref(c);
}

TEST_F(DynamicSamePaddingFusionTests, PreserveTensorNames) {
    comparator.enable(FunctionsComparator::TENSOR_NAMES);
    model = get_model(Config{});
    model_ref = get_model_ref(Config{});
}

TEST_F(DynamicSamePaddingFusionTests, UnknownKernelShape) {
    model = get_model(Config{});
    const auto weights = std::make_shared<Parameter>(element::f32, PartialShape{4, 3, -1, -1});
    model->get_results()[0]->input_value(0).get_node_shared_ptr()->set_argument(1, weights);
    model->add_parameters({weights});
    model->validate_nodes_and_infer_types();
}

TEST_F(DynamicSamePaddingFusionTests, UnknownInputRank) {
    Config c;
    c.shape = PartialShape::dynamic();
    model = get_model(c);
}

TEST_F(DynamicSamePaddingFusionTests, StaticOddEvenInputsAccuracy) {
    Config c;
    c.shape = {1, 3, 7, 8};
    c.kernel = {2, 5};
    c.dilations = {2, 1};
    comparator.enable(FunctionsComparator::ACCURACY);
    model = get_model(c);
    model_ref = get_model_ref(c);
}

TEST_F(DynamicSamePaddingFusionTests, InputSmallerThanEffectiveKernelAccuracy) {
    Config c;
    c.shape = {1, 3, 1, 2};
    c.grouped = true;
    c.dilations = {2, 2};
    comparator.enable(FunctionsComparator::ACCURACY);
    model = get_model(c);
    model_ref = get_model_ref(c);
}

TEST_F(DynamicSamePaddingFusionTests, NonzeroPaddingValue) {
    Config c;
    c.pad_value = 1;
    model = get_model(c);
}

TEST_F(DynamicSamePaddingFusionTests, EdgePadding) {
    Config c;
    c.pad_mode = op::PadMode::EDGE;
    model = get_model(c);
}

TEST_F(DynamicSamePaddingFusionTests, NonSpatialPadding) {
    Config c;
    c.batch_pad = 1;
    model = get_model(c);
}

TEST_F(DynamicSamePaddingFusionTests, ExistingConvolutionPadding) {
    Config c;
    c.conv_pads = {1, 0};
    model = get_model(c);
}

TEST_F(DynamicSamePaddingFusionTests, ExistingAutoPadding) {
    Config c;
    c.auto_pad = op::PadType::SAME_UPPER;
    model = get_model(c);
}

TEST_F(DynamicSamePaddingFusionTests, DifferentShapeSource) {
    Config c;
    c.wrong_source = true;
    model = get_model(c);
}

TEST_F(DynamicSamePaddingFusionTests, SwappedSpatialAxes) {
    Config c;
    c.wrong_axis = true;
    model = get_model(c);
}

TEST_F(DynamicSamePaddingFusionTests, DifferentEffectiveKernel) {
    Config c;
    c.kernel_adjustment = 1;
    model = get_model(c);
}

TEST_F(DynamicSamePaddingFusionTests, DifferentPaddingSplit) {
    Config c;
    c.split_divisor = 3;
    model = get_model(c);
}

TEST_F(DynamicSamePaddingFusionTests, SameLowerPadding) {
    Config c;
    c.same_lower = true;
    model = get_model(c);
}

TEST_F(DynamicSamePaddingFusionTests, LowPrecisionShapeArithmetic) {
    Config c;
    c.math_type = element::f16;
    model = get_model(c);
}

TEST_F(DynamicSamePaddingFusionTests, RoundedReciprocalStride) {
    Config c;
    c.math_type = element::f64;
    c.strides = {3, 2};
    c.reciprocal_stride = true;
    model = get_model(c);
}

TEST_F(DynamicSamePaddingFusionTests, TransformationCallback) {
    manager.get_pass_config()->set_callback<pass::DynamicSamePaddingFusion>([](const std::shared_ptr<const Node>&) {
        return true;
    });
    model = get_model(Config{});
}

TEST_F(TransformationTestsF, DynamicSamePaddingFusionInMOCTransformations) {
    const Config c;
    model = get_model(c);
    model_ref = get_model_ref(c);
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CONST_VALUES);
    manager.register_pass<pass::MOCTransformations>(true, false);
}
}  // namespace
