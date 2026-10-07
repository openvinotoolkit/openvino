// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/dynamic_same_padding_fusion.hpp"

#include <gtest/gtest.h>

#include <algorithm>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/convolution.hpp"
#include "intel_gpu/plugin/transformations_pipeline.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/runtime/properties.hpp"
#include "test_utils.h"
#include "transformations/common_optimizations/common_optimizations.hpp"
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
    bool omit_zero_offset = false;
    bool constant_fold_decrement = false;
    bool cancel_dimension = false;
    bool cancel_end = false;
    double cancellation_offset = 16777216;
    double dimension_scale = 1;
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
        const auto size = std::make_shared<Convert>(std::make_shared<Gather>(shape, Constant::create(element::i64, Shape{}, {axis}), axis_zero), c.math_type);
        const auto stride_value = static_cast<double>(c.strides[i]);
        const auto stride = scalar_constant(c, stride_value);
        Output<Node> divided;
        if (c.reciprocal_stride)
            divided = std::make_shared<Multiply>(size, scalar_constant(c, 1.0 / stride_value));
        else
            divided = std::make_shared<Divide>(size, stride);
        const auto ceil = std::make_shared<Ceiling>(divided);
        const auto effective = static_cast<double>((c.kernel[i] - 1) * c.dilations[i] + 1) + c.kernel_adjustment;
        Output<Node> extent;
        if (c.expanded) {
            const Output<Node> decrement =
                c.constant_fold_decrement ? Output<Node>(std::make_shared<Add>(ceil, scalar_constant(c, -1))) : subtract(ceil, scalar_constant(c, 1), c);
            extent = std::make_shared<Add>(std::make_shared<Multiply>(decrement, stride), scalar_constant(c, effective));
        } else {
            extent = std::make_shared<Multiply>(stride, ceil);
            if (!c.omit_zero_offset || effective != stride_value)
                extent = std::make_shared<Add>(extent, scalar_constant(c, effective - stride_value));
        }
        Output<Node> dimension = size;
        if (c.cancel_dimension) {
            const auto offset = scalar_constant(c, c.cancellation_offset);
            dimension = subtract(std::make_shared<Add>(dimension, offset), offset, c);
        }
        if (c.dimension_scale != 1) {
            dimension = std::make_shared<Multiply>(std::make_shared<Multiply>(dimension, scalar_constant(c, c.dimension_scale)),
                                                   scalar_constant(c, 1.0 / c.dimension_scale));
        }
        const auto total = std::make_shared<Maximum>(scalar_constant(c, 0), subtract(extent, dimension, c));
        Output<Node> half;
        if (c.reciprocal)
            half = std::make_shared<Multiply>(scalar_constant(c, 1 / c.split_divisor), total);
        else
            half = std::make_shared<Divide>(total, scalar_constant(c, c.split_divisor));
        half = std::make_shared<Floor>(half);
        Output<Node> before = std::make_shared<Convert>(half, c.shape_type);
        Output<Node> end_total = total;
        if (c.cancel_end) {
            const auto offset = scalar_constant(c, c.cancellation_offset);
            end_total = subtract(std::make_shared<Add>(end_total, offset), offset, c);
        }
        Output<Node> after = std::make_shared<Convert>(subtract(end_total, half, c), c.shape_type);
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
        conv = std::make_shared<GroupConvolution>(data, weights, c.strides, c.conv_pads, c.conv_pads, c.dilations, auto_pad);
    else
        conv = std::make_shared<Convolution>(data, weights, c.strides, c.conv_pads, c.conv_pads, c.dilations, auto_pad);
    conv->set_friendly_name(name);
    conv->output(0).get_tensor().set_names({name + "_output"});
    return conv;
}

Output<Node> spatial_shape(const Output<Node>& shape) {
    return std::make_shared<Gather>(shape, Constant::create(element::i64, Shape{2}, {2, 3}), Constant::create(element::i64, Shape{}, {0}));
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
        manager.register_pass<intel_gpu::DynamicSamePaddingFusion>();
    }
};

using TestParams = std::tuple<bool, bool, bool, size_t>;
class DynamicSamePaddingFusionParameterized : public DynamicSamePaddingFusionTests, public testing::WithParamInterface<TestParams> {
public:
    static std::string get_test_name(const testing::TestParamInfo<TestParams>& info) {
        const auto& [grouped, pad_v1, expanded, spatial_rank] = info.param;
        return std::string(grouped ? "GroupConv" : "Conv") + (pad_v1 ? "_Pad1" : "_Pad12") + (expanded ? "_Expanded" : "_Folded") + "_" +
               std::to_string(spatial_rank) + "D";
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

TEST_F(DynamicSamePaddingFusionTests, FoldedZeroOffsetWithoutAdd) {
    Config c;
    c.kernel = {2, 2};
    c.omit_zero_offset = true;
    model = get_model(c);
    model_ref = get_model_ref(c);
}

TEST_F(DynamicSamePaddingFusionTests, ExpandedConstantFoldedDecrement) {
    Config c;
    c.expanded = true;
    c.constant_fold_decrement = true;
    c.decompose_subtract = true;
    model = get_model(c);
    model_ref = get_model_ref(c);
}

TEST_F(DynamicSamePaddingFusionTests, FoldedNegativeOffset) {
    Config c;
    c.kernel = {1, 1};
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

using CancellationParams = std::tuple<bool, bool, bool>;
class DynamicSamePaddingCancellationTests : public DynamicSamePaddingFusionTests, public testing::WithParamInterface<CancellationParams> {
public:
    static std::string get_test_name(const testing::TestParamInfo<CancellationParams>& info) {
        const auto& [cancel_end, decompose_subtract, use_f64] = info.param;
        return std::string(cancel_end ? "End" : "Dimension") + (decompose_subtract ? "_AddNegate" : "_Subtract") + (use_f64 ? "_f64" : "_f32");
    }
};

TEST_P(DynamicSamePaddingCancellationTests, PreserveFloatCancellation) {
    const auto& [cancel_end, decompose_subtract, use_f64] = GetParam();
    Config c;
    c.cancel_end = cancel_end;
    c.cancel_dimension = !cancel_end;
    c.decompose_subtract = decompose_subtract;
    c.math_type = use_f64 ? element::f64 : element::f32;
    c.cancellation_offset = use_f64 ? 9007199254740992.0 : 16777216.0;
    model = get_model(c);
    if (!use_f64) {
        // Evaluate the original shape arithmetic directly. Compiling a dynamic
        // reference could run transformations and mask the incorrect fusion.
        const auto conv = model->get_results()[0]->input_value(0).get_node_shared_ptr();
        const auto pad = conv->input_value(0).get_node_shared_ptr();
        const auto pads = std::make_shared<Model>(OutputVector{pad->input_value(1), pad->input_value(2)}, model->get_parameters());
        const size_t length = cancel_end ? 6 : 7;
        const TensorVector inputs{Tensor(element::f32, Shape{1, 3, length, length})};
        TensorVector outputs{Tensor(element::i64, Shape{4}), Tensor(element::i64, Shape{4})};
        ASSERT_TRUE(pads->evaluate(outputs, inputs));
        for (size_t axis = 2; axis < 4; ++axis) {
            EXPECT_EQ(outputs[0].data<const int64_t>()[axis], 0);
            EXPECT_EQ(outputs[1].data<const int64_t>()[axis], cancel_end ? 0 : 1);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         DynamicSamePaddingCancellationTests,
                         testing::Combine(testing::Bool(), testing::Bool(), testing::Bool()),
                         DynamicSamePaddingCancellationTests::get_test_name);

TEST_F(DynamicSamePaddingFusionTests, PreserveDimensionScaling) {
    Config c;
    c.dimension_scale = 0.1;
    model = get_model(c);
}

TEST_F(DynamicSamePaddingFusionTests, TransformationCallback) {
    manager.get_pass_config()->set_callback<intel_gpu::DynamicSamePaddingFusion>([](const std::shared_ptr<const Node>&) {
        return true;
    });
    model = get_model(Config{});
}

TEST_F(TransformationTestsF, DynamicSamePaddingFusionBeforeCommonOptimizations) {
    const Config c;
    model = get_model(c);
    model_ref = get_model_ref(c);
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CONST_VALUES);
    manager.register_pass<intel_gpu::DynamicSamePaddingFusion>();
    manager.register_pass<pass::CommonOptimizations>();
}

TEST_F(TransformationTestsF, DynamicSamePaddingFusionZeroOffsetBeforeCommonOptimizations) {
    Config c;
    c.kernel = {2, 2};
    model = get_model(c);
    model_ref = get_model_ref(c);
    comparator.enable(FunctionsComparator::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CONST_VALUES);
    manager.register_pass<intel_gpu::DynamicSamePaddingFusion>();
    manager.register_pass<pass::CommonOptimizations>();
}

TEST(DynamicSamePaddingFusionRegistration, NotInMOCTransformations) {
    const auto model = get_model(Config{});
    pass::MOCTransformations moc(true, false);
    moc.run_on_model(model);
    const auto conv = as_type_ptr<op::v1::Convolution>(model->get_results()[0]->input_value(0).get_node_shared_ptr());
    ASSERT_NE(conv, nullptr);
    EXPECT_EQ(conv->get_auto_pad(), op::PadType::EXPLICIT);
    EXPECT_TRUE(is_type<op::util::PadBase>(conv->input_value(0).get_node()));
}

using PipelineParams = std::tuple<bool, bool, size_t, element::Type>;
class DynamicSamePaddingFusionPipelineTests : public testing::TestWithParam<PipelineParams> {
public:
    static std::string get_test_name(const testing::TestParamInfo<PipelineParams>& info) {
        const auto& [grouped, expanded, spatial_rank, precision] = info.param;
        return std::string(grouped ? "GroupConv" : "Conv") + (expanded ? "_Expanded" : "_Folded") + "_" + std::to_string(spatial_rank) + "D_" +
               precision.get_type_name();
    }
};

TEST_P(DynamicSamePaddingFusionPipelineTests, FuseBeforeConvolutionLowering) {
    const auto& [grouped, expanded, spatial_rank, precision] = GetParam();
    Config c;
    c.grouped = grouped;
    c.expanded = expanded;
    c.shape = PartialShape::dynamic(spatial_rank + 2);
    c.shape[1] = 3;
    c.kernel.assign(spatial_rank, 3);
    c.strides.assign(spatial_rank, 2);
    c.dilations.assign(spatial_rank, 1);
    c.conv_pads.assign(spatial_rank, 0);
    const auto model = get_model(c);

    auto& engine = tests::get_test_engine();
    auto context = std::make_shared<intel_gpu::RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto config = tests::get_test_default_config(engine);
    config.set_user_property(ov::hint::inference_precision(precision));
    config.finalize(context.get(), model.get());
    intel_gpu::TransformationsPipeline pipeline(config, context);
    pipeline.apply(model);

    const auto ops = model->get_ordered_ops();
    EXPECT_TRUE(std::none_of(ops.begin(), ops.end(), [](const auto& node) {
        return is_type<op::util::PadBase>(node);
    }));
    const auto conv = std::find_if(ops.begin(), ops.end(), [](const auto& node) {
        return is_type<intel_gpu::op::Convolution>(node);
    });
    ASSERT_NE(conv, ops.end());
    const auto internal_conv = as_type_ptr<intel_gpu::op::Convolution>(*conv);
    EXPECT_EQ(internal_conv->get_auto_pad(), op::PadType::SAME_UPPER);
    EXPECT_EQ(internal_conv->get_strides(), c.strides);
    EXPECT_EQ(internal_conv->get_dilations(), c.dilations);
    EXPECT_EQ(internal_conv->get_output_element_type(0), precision);
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         DynamicSamePaddingFusionPipelineTests,
                         testing::Combine(testing::Bool(), testing::Bool(), testing::Values(1, 2, 3), testing::Values(element::f32, element::f16)),
                         DynamicSamePaddingFusionPipelineTests::get_test_name);
}  // namespace
