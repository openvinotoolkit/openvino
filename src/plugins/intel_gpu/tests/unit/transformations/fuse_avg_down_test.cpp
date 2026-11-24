// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/fuse_avg_down.hpp"

#include <gtest/gtest.h>

#include <string>
#include <tuple>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/grouped_space_to_depth.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"

namespace {

std::shared_ptr<ov::op::v0::Constant> shape_part(const std::vector<int64_t>& values) {
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
}

std::shared_ptr<ov::op::v0::Constant> scalar(int64_t value) {
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {value});
}

enum class DynamicPaddingVariant { Canonical, MisplacedTemporal, NonzeroSpatial };

// Describes one AvgDown subgraph: data -> [Pad] -> Reshape -> Transpose -> [Reshape] -> Reshape -> ReduceMean
struct AvgDownParams {
    int64_t input_channels = 96;
    int64_t output_channels = 96;
    int64_t factor_t = 1;
    int64_t factor_s = 2;
    int64_t time = 5;
    int64_t height = 8;
    int64_t width = 10;
    bool add_pad = false;
    float pad_value = 0.0f;             // fill of the temporal Pad
    bool add_consumer = false;          // Relu after ReduceMean
    bool fold_reshapes = false;         // no flatten Reshape between Transpose and the group Reshape
    bool special_zero = false;          // special_zero on every chain Reshape
    bool dynamic_temporal_pad = false;  // dynamic input dims, pads computed from ShapeOf
    bool dynamic_batch = false;         // batch of the target shapes taken from ShapeOf
    DynamicPaddingVariant padding_variant = DynamicPaddingVariant::Canonical;
    std::vector<int64_t> transpose_order = {0, 1, 3, 5, 7, 2, 4, 6};
    int64_t reduce_axis = 2;  // group axis of the 6-D view; -4 is the same axis
};

std::shared_ptr<ov::op::v0::Parameter> make_input(const AvgDownParams& p) {
    const auto dim = [&](int64_t value) {
        return p.dynamic_temporal_pad ? ov::Dimension::dynamic() : ov::Dimension(value);
    };
    return std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{dim(1), p.input_channels, dim(p.time), dim(p.height), dim(p.width)});
}

ov::Output<ov::Node> make_dynamic_pads_begin(const ov::Output<ov::Node>& data, const AvgDownParams& p) {
    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(data, ov::element::i64);
    auto time_length = std::make_shared<ov::op::v8::Gather>(shape_of, shape_part({2}), scalar(0));
    auto inner_mod = std::make_shared<ov::op::v1::FloorMod>(time_length, shape_part({p.factor_t}));
    auto negated_mod = std::make_shared<ov::op::v1::Multiply>(inner_mod, shape_part({-1}));
    auto difference = std::make_shared<ov::op::v1::Add>(shape_part({p.factor_t}), negated_mod);
    auto outer_mod = std::make_shared<ov::op::v1::FloorMod>(difference, shape_part({p.factor_t}));
    ov::OutputVector padding_parts = {shape_part({0}), shape_part({0}), shape_part({0}), shape_part({0}), outer_mod, shape_part({0})};
    if (p.padding_variant == DynamicPaddingVariant::MisplacedTemporal) {
        std::swap(padding_parts[0], padding_parts[4]);
    } else if (p.padding_variant == DynamicPaddingVariant::NonzeroSpatial) {
        padding_parts[0] = shape_part({1});
    }
    auto padding_values = std::make_shared<ov::op::v0::Concat>(padding_parts, 0);
    auto paired_padding = std::make_shared<ov::op::v1::Reshape>(padding_values, shape_part({-1, 2}), false);
    auto split_padding = std::make_shared<ov::op::v1::Split>(paired_padding, scalar(1), 2);
    auto first_column = std::make_shared<ov::op::v0::Squeeze>(split_padding->output(0), shape_part({1}));
    auto reversed_padding = std::make_shared<ov::op::v8::Gather>(first_column, shape_part({2, 1, 0}), scalar(0));
    return std::make_shared<ov::op::v0::Concat>(ov::OutputVector{shape_part({0, 0}), reversed_padding}, 0);
}

std::shared_ptr<ov::Model> getModel(const AvgDownParams& p) {
    const int64_t pad_begin_t = (p.factor_t - p.time % p.factor_t) % p.factor_t;
    const int64_t output_time = (p.time + pad_begin_t) / p.factor_t;
    const int64_t factor_volume = p.factor_t * p.factor_s * p.factor_s;
    const int64_t group_size = p.input_channels * factor_volume / p.output_channels;

    auto input = make_input(p);
    ov::Output<ov::Node> data = input;
    if (p.add_pad) {
        auto pads_begin = p.dynamic_temporal_pad ? make_dynamic_pads_begin(data, p) : shape_part({0, 0, pad_begin_t, 0, 0});
        data = std::make_shared<ov::op::v12::Pad>(data,
                                                  pads_begin,
                                                  shape_part({0, 0, 0, 0, 0}),
                                                  ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {p.pad_value}),
                                                  ov::op::PadMode::CONSTANT);
    }

    ov::Output<ov::Node> batch = shape_part({1});
    if (p.dynamic_batch) {
        auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(data, ov::element::i64);
        batch = std::make_shared<ov::op::v8::Gather>(shape_of, shape_part({0}), scalar(0));
    }
    auto batch_channels = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{batch, shape_part({p.input_channels})}, 0);
    auto time = shape_part({output_time});
    auto height = shape_part({p.height / p.factor_s});
    auto width = shape_part({p.width / p.factor_s});
    auto factor_shape = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{batch_channels, time, shape_part({p.factor_t}), height, shape_part({p.factor_s}), width, shape_part({p.factor_s})},
        0);
    auto factor_reshape = std::make_shared<ov::op::v1::Reshape>(data, factor_shape, p.special_zero);
    auto transpose = std::make_shared<ov::op::v1::Transpose>(factor_reshape, shape_part(p.transpose_order));

    ov::Output<ov::Node> group_input = transpose;
    if (!p.fold_reshapes) {
        ov::OutputVector flatten_parts;
        if (factor_volume == 1) {
            flatten_parts = {batch_channels, time, height, width};
        } else {
            flatten_parts = {batch, shape_part({p.input_channels * factor_volume}), time, height, width};
        }
        auto flatten_shape = std::make_shared<ov::op::v0::Concat>(flatten_parts, 0);
        group_input = std::make_shared<ov::op::v1::Reshape>(transpose, flatten_shape, p.special_zero);
    }
    auto group_shape =
        std::make_shared<ov::op::v0::Concat>(ov::OutputVector{batch, shape_part({p.output_channels}), shape_part({group_size}), time, height, width}, 0);
    auto group_reshape = std::make_shared<ov::op::v1::Reshape>(group_input, group_shape, p.special_zero);
    ov::Output<ov::Node> output = std::make_shared<ov::op::v1::ReduceMean>(group_reshape, shape_part({p.reduce_axis}), false);
    if (p.add_consumer) {
        output = std::make_shared<ov::op::v0::Relu>(output);
    }
    return std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{input});
}

std::shared_ptr<ov::Model> getModelRef(const AvgDownParams& p) {
    auto input = make_input(p);
    ov::Output<ov::Node> output = input;
    if (p.factor_t == 2 && p.factor_s == 2) {
        output = std::make_shared<ov::intel_gpu::op::GroupedSpaceToDepth>(input, 2, 2, static_cast<size_t>(p.output_channels));
    } else if (p.factor_t == 1 && p.factor_s == 2) {
        output = std::make_shared<ov::op::v1::AvgPool>(input,
                                                       ov::Strides{1, 2, 2},
                                                       ov::Shape{0, 0, 0},
                                                       ov::Shape{0, 0, 0},
                                                       ov::Shape{1, 2, 2},
                                                       false,
                                                       ov::op::RoundingType::FLOOR,
                                                       ov::op::PadType::EXPLICIT);
    }
    if (p.add_consumer) {
        output = std::make_shared<ov::op::v0::Relu>(output);
    }
    return std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{input});
}

// Hand-written chain with literal Reshape target shapes: input -> Reshape(factor_shape) -> Transpose ->
// [Reshape(flatten_shape)] -> Reshape(group_shape) -> ReduceMean -> Relu. Used to build reshapes that do not split
// the input as AvgDown does, and special_zero reshapes with a literal 0 ("copy this dimension") in the target shapes.
std::shared_ptr<ov::Model> getLiteralChainModel(const ov::Shape& input_shape,
                                                const std::vector<int64_t>& factor_shape,
                                                const std::vector<int64_t>& group_shape,
                                                bool special_zero = false,
                                                const std::vector<int64_t>& flatten_shape = {}) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    auto factor_reshape = std::make_shared<ov::op::v1::Reshape>(input, shape_part(factor_shape), special_zero);
    auto transpose = std::make_shared<ov::op::v1::Transpose>(factor_reshape, shape_part({0, 1, 3, 5, 7, 2, 4, 6}));
    ov::Output<ov::Node> group_input = transpose;
    if (!flatten_shape.empty()) {
        group_input = std::make_shared<ov::op::v1::Reshape>(transpose, shape_part(flatten_shape), special_zero);
    }
    auto group_reshape = std::make_shared<ov::op::v1::Reshape>(group_input, shape_part(group_shape), special_zero);
    auto reduce = std::make_shared<ov::op::v1::ReduceMean>(group_reshape, shape_part({2}), false);
    auto relu = std::make_shared<ov::op::v0::Relu>(reduce);
    return std::make_shared<ov::Model>(ov::OutputVector{relu}, ov::ParameterVector{input});
}

std::shared_ptr<ov::op::v12::Pad> find_pad(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ops()) {
        if (const auto pad = ov::as_type_ptr<ov::op::v12::Pad>(node)) {
            return pad;
        }
    }
    return {};
}

std::shared_ptr<ov::op::v1::ReduceMean> find_reduce(const std::shared_ptr<ov::Model>& model) {
    for (const auto& node : model->get_ops()) {
        if (const auto reduce = ov::as_type_ptr<ov::op::v1::ReduceMean>(node)) {
            return reduce;
        }
    }
    return {};
}

// Positive cases with static shapes: the canonical 5-node form and the folded form (flatten Reshape merged
// into the group Reshape), with and without special_zero.
// The "C96to96_ft1_fs2_canonical" case is the former RewritesSpatialAverageToAvgPool3D test, and the
// "C{96to192,192to384,384to768}_ft2_fs2_canonical" cases are the former
// RewritesAllTemporalGroupedBlocksToGroupedSpaceToDepth test.
using FuseAvgDownStaticParams = std::tuple<int64_t,  // input channels
                                           int64_t,  // output channels
                                           int64_t,  // factor_t
                                           int64_t,  // factor_s
                                           bool,     // fold reshapes
                                           bool,     // special_zero
                                           bool>;    // Relu consumer after ReduceMean

class FuseAvgDownStaticTests : public TransformationTestsF, public testing::WithParamInterface<FuseAvgDownStaticParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FuseAvgDownStaticParams>& obj) {
        const auto& [input_channels, output_channels, factor_t, factor_s, fold_reshapes, special_zero, add_consumer] = obj.param;
        return "C" + std::to_string(input_channels) + "to" + std::to_string(output_channels) + "_ft" + std::to_string(factor_t) + "_fs" +
               std::to_string(factor_s) + (fold_reshapes ? "_folded" : "_canonical") + (special_zero ? "_special_zero" : "") +
               (add_consumer ? "_consumer" : "");
    }
};

TEST_P(FuseAvgDownStaticTests, Rewrites) {
    const auto& [input_channels, output_channels, factor_t, factor_s, fold_reshapes, special_zero, add_consumer] = GetParam();
    AvgDownParams p;
    p.input_channels = input_channels;
    p.output_channels = output_channels;
    p.factor_t = factor_t;
    p.factor_s = factor_s;
    p.add_pad = factor_t > 1;
    p.add_consumer = add_consumer;
    p.fold_reshapes = fold_reshapes;
    p.special_zero = special_zero;
    model = getModel(p);
    model_ref = getModelRef(p);
    manager.register_pass<ov::intel_gpu::FuseAvgDown>();
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    // GroupedSpaceToDepth is a plugin op without a reference evaluate()
    if (factor_t == 1) {
        comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    }
}

// (input channels, output channels, factor_t, factor_s) of the supported downsample kinds
const std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> avg_down_kinds = {
    {96, 96, 1, 1},    // identity
    {96, 96, 1, 2},    // spatial -> AvgPool
    {96, 192, 2, 2},   // temporal/spatial -> GroupedSpaceToDepth
    {192, 384, 2, 2},  // temporal/spatial -> GroupedSpaceToDepth
    {384, 768, 2, 2},  // temporal/spatial -> GroupedSpaceToDepth
};

std::vector<FuseAvgDownStaticParams> make_static_params() {
    std::vector<FuseAvgDownStaticParams> params;
    for (const auto& [input_channels, output_channels, factor_t, factor_s] : avg_down_kinds) {
        for (const bool fold_reshapes : {false, true}) {
            for (const bool special_zero : {false, true}) {
                for (const bool add_consumer : {false, true}) {
                    // identity rewrite: with the Result as the only consumer, replace_output_update_name() moves the output name
                    // onto the Parameter, which is outside of what the reference model can express; covered with a Relu consumer
                    if (factor_s == 1 && !add_consumer) {
                        continue;
                    }
                    params.emplace_back(input_channels, output_channels, factor_t, factor_s, fold_reshapes, special_zero, add_consumer);
                }
            }
        }
    }
    return params;
}

INSTANTIATE_TEST_SUITE_P(smoke_FuseAvgDown, FuseAvgDownStaticTests, testing::ValuesIn(make_static_params()), FuseAvgDownStaticTests::getTestCaseName);

// encoder/downsamples.{0,1,2}/avg_shortcut of a torch exported video VAE encoder: folded Reshapes,
// special_zero=true and a temporal Pad that only aligns the frame count.
using FuseAvgDownSiteParams = std::tuple<int64_t,  // input channels
                                         int64_t,  // output channels
                                         int64_t,  // factor_t
                                         int64_t,  // time
                                         int64_t,  // height
                                         int64_t,  // width
                                         bool>;    // temporal Pad in front of the chain

class FuseAvgDownTorchSiteTests : public TransformationTestsF, public testing::WithParamInterface<FuseAvgDownSiteParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<FuseAvgDownSiteParams>& obj) {
        const auto& [input_channels, output_channels, factor_t, time, height, width, add_pad] = obj.param;
        return "C" + std::to_string(input_channels) + "to" + std::to_string(output_channels) + "_ft" + std::to_string(factor_t) + "_T" + std::to_string(time) +
               "_H" + std::to_string(height) + "_W" + std::to_string(width) + (add_pad ? "_pad" : "");
    }
};

TEST_P(FuseAvgDownTorchSiteTests, Rewrites) {
    const auto& [input_channels, output_channels, factor_t, time, height, width, add_pad] = GetParam();
    AvgDownParams p;
    p.input_channels = input_channels;
    p.output_channels = output_channels;
    p.factor_t = factor_t;
    p.factor_s = 2;
    p.time = time;
    p.height = height;
    p.width = width;
    p.add_pad = add_pad;
    p.fold_reshapes = true;
    p.special_zero = true;
    model = getModel(p);
    model_ref = getModelRef(p);
    manager.register_pass<ov::intel_gpu::FuseAvgDown>();
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

INSTANTIATE_TEST_SUITE_P(smoke_FuseAvgDown,
                         FuseAvgDownTorchSiteTests,
                         testing::Values(FuseAvgDownSiteParams{160, 160, 1, 1, 112, 224, false},
                                         FuseAvgDownSiteParams{160, 160, 1, 1, 112, 224, true},
                                         FuseAvgDownSiteParams{160, 320, 2, 1, 56, 112, true},
                                         FuseAvgDownSiteParams{320, 640, 2, 1, 28, 56, true}),
                         FuseAvgDownTorchSiteTests::getTestCaseName);

struct SpecialZeroLiteralCase {
    std::string name;
    ov::Shape input_shape;
    std::vector<int64_t> factor_shape;
    std::vector<int64_t> flatten_shape;  // empty: folded chain without the flatten Reshape
    std::vector<int64_t> group_shape;
    int64_t output_channels;
    int64_t factor_t;
};

class FuseAvgDownSpecialZeroLiteralTests : public TransformationTestsF, public testing::WithParamInterface<SpecialZeroLiteralCase> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SpecialZeroLiteralCase>& obj) {
        return obj.param.name;
    }
};

TEST_P(FuseAvgDownSpecialZeroLiteralTests, Rewrites) {
    const auto& c = GetParam();
    model = getLiteralChainModel(c.input_shape, c.factor_shape, c.group_shape, true, c.flatten_shape);
    AvgDownParams p;
    p.input_channels = static_cast<int64_t>(c.input_shape[1]);
    p.output_channels = c.output_channels;
    p.factor_t = c.factor_t;
    p.factor_s = 2;
    p.time = static_cast<int64_t>(c.input_shape[2]);
    p.height = static_cast<int64_t>(c.input_shape[3]);
    p.width = static_cast<int64_t>(c.input_shape[4]);
    p.add_consumer = true;
    model_ref = getModelRef(p);
    manager.register_pass<ov::intel_gpu::FuseAvgDown>();
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    // GroupedSpaceToDepth is a plugin op without a reference evaluate()
    if (c.factor_t == 1) {
        comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    }
}

INSTANTIATE_TEST_SUITE_P(
    smoke_FuseAvgDown,
    FuseAvgDownSpecialZeroLiteralTests,
    testing::Values(
        SpecialZeroLiteralCase{"C96to192_ft2_fs2_zero_factors_canonical",
                               {1, 96, 2, 2, 2},
                               {0, 0, 1, 0, 1, 2, 1, 2},
                               {0, 768, 1, 1, 1},
                               {0, 192, 4, 1, 1, 1},
                               192,
                               2},
        SpecialZeroLiteralCase{"C96to192_ft2_fs2_zero_factors_folded", {1, 96, 2, 2, 2}, {0, 0, 1, 0, 1, 2, 1, 2}, {}, {0, 192, 4, 1, 1, 1}, 192, 2},
        SpecialZeroLiteralCase{"C96to192_ft2_fs2_zero_factor_t_canonical",
                               {1, 96, 4, 2, 6},
                               {0, 0, 2, 0, 1, 2, 3, 2},
                               {0, 768, 2, 1, 3},
                               {0, 192, 4, 2, 1, 3},
                               192,
                               2},
        SpecialZeroLiteralCase{"C96to192_ft2_fs2_zero_factor_t_folded", {1, 96, 4, 2, 6}, {0, 0, 2, 0, 1, 2, 3, 2}, {}, {0, 192, 4, 2, 1, 3}, 192, 2},
        SpecialZeroLiteralCase{"C96to96_ft1_fs2_zero_dims_canonical",
                               {1, 96, 1, 2, 2},
                               {0, 0, 0, 1, 1, 2, 1, 2},
                               {0, 384, 1, 1, 1},
                               {0, 96, 4, 1, 1, 1},
                               96,
                               1},
        SpecialZeroLiteralCase{"C96to96_ft1_fs2_zero_dims_folded", {1, 96, 1, 2, 2}, {0, 0, 0, 1, 1, 2, 1, 2}, {}, {0, 96, 4, 1, 1, 1}, 96, 1}),
    FuseAvgDownSpecialZeroLiteralTests::getTestCaseName);

TEST_F(TransformationTestsF, FuseAvgDownRemovesUnitFactorChainAndPadding) {
    AvgDownParams p;
    p.input_channels = 768;
    p.output_channels = 768;
    p.factor_s = 1;
    p.add_pad = true;
    p.add_consumer = true;
    model = getModel(p);
    model_ref = getModelRef(p);
    manager.register_pass<ov::intel_gpu::FuseAvgDown>();
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

// Former RewritesDynamicTemporalPadsWithStaticSourceChannels test: the Pad output channels are dynamic, so the
// factors come from the shape-expression contract.
class FuseAvgDownDynamicPadTests : public TransformationTestsF, public testing::WithParamInterface<std::tuple<int64_t, int64_t, int64_t>> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::tuple<int64_t, int64_t, int64_t>>& obj) {
        const auto& [input_channels, output_channels, factor] = obj.param;
        return "C" + std::to_string(input_channels) + "to" + std::to_string(output_channels) + "_ft" + std::to_string(factor) + "_fs" + std::to_string(factor);
    }
};

TEST_P(FuseAvgDownDynamicPadTests, Rewrites) {
    const auto& [input_channels, output_channels, factor_t] = GetParam();
    AvgDownParams p;
    p.input_channels = input_channels;
    p.output_channels = output_channels;
    p.factor_t = factor_t;
    p.factor_s = factor_t;
    p.add_pad = true;
    p.add_consumer = factor_t == 1;
    p.dynamic_temporal_pad = true;
    model = getModel(p);
    const auto pad = find_pad(model);
    ASSERT_TRUE(pad);
    pad->set_output_type(0, pad->get_output_element_type(0), ov::PartialShape::dynamic(5));
    ASSERT_TRUE(pad->get_output_partial_shape(0)[1].is_dynamic());
    model_ref = getModelRef(p);
    manager.register_pass<ov::intel_gpu::FuseAvgDown>();
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

INSTANTIATE_TEST_SUITE_P(smoke_FuseAvgDown,
                         FuseAvgDownDynamicPadTests,
                         testing::Values(std::make_tuple(96, 192, 2), std::make_tuple(192, 384, 2), std::make_tuple(384, 768, 2), std::make_tuple(768, 768, 1)),
                         FuseAvgDownDynamicPadTests::getTestCaseName);

TEST_F(TransformationTestsF, FuseAvgDownRewritesDynamicBatchWithoutSpecialZero) {
    // Dynamic reshape outputs without special_zero: the factors come from the shape-expression contract
    AvgDownParams p;
    p.dynamic_temporal_pad = true;
    p.dynamic_batch = true;
    model = getModel(p);
    const auto reduce = find_reduce(model);
    ASSERT_TRUE(reduce);
    ASSERT_TRUE(reduce->get_input_partial_shape(0).is_dynamic());
    model_ref = getModelRef(p);
    manager.register_pass<ov::intel_gpu::FuseAvgDown>();
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

// Negative cases: model_ref stays nullptr, so the model is expected to stay unchanged.

TEST_F(TransformationTestsF, FuseAvgDownRejectsSpecialZeroWithDynamicShapes) {
    // With special_zero a literal 0 in a target shape means "copy this dimension", so while the reshape
    // outputs stay dynamic the factors cannot be recovered from the shape expressions
    AvgDownParams p;
    p.dynamic_temporal_pad = true;
    p.dynamic_batch = true;
    p.special_zero = true;
    model = getModel(p);
    const auto reduce = find_reduce(model);
    ASSERT_TRUE(reduce);
    ASSERT_TRUE(reduce->get_input_partial_shape(0).is_dynamic());
    manager.register_pass<ov::intel_gpu::FuseAvgDown>();
}

struct NonzeroPadFillCase {
    std::string name;
    float pad_value;
};

class FuseAvgDownRejectsNonzeroPadFillTests : public TransformationTestsF, public testing::WithParamInterface<NonzeroPadFillCase> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<NonzeroPadFillCase>& obj) {
        return obj.param.name;
    }
};

TEST_P(FuseAvgDownRejectsNonzeroPadFillTests, Rejects) {
    AvgDownParams p;
    p.input_channels = 160;
    p.output_channels = 320;
    p.factor_t = 2;
    p.time = 1;  // one padded frame in front
    p.height = 56;
    p.width = 112;
    p.add_pad = true;
    p.pad_value = GetParam().pad_value;
    p.fold_reshapes = true;
    p.special_zero = true;
    model = getModel(p);
    ASSERT_TRUE(find_pad(model));
    manager.register_pass<ov::intel_gpu::FuseAvgDown>();
}

INSTANTIATE_TEST_SUITE_P(smoke_FuseAvgDown,
                         FuseAvgDownRejectsNonzeroPadFillTests,
                         testing::Values(NonzeroPadFillCase{"fill_0p5", 0.5f}, NonzeroPadFillCase{"fill_m0p25", -0.25f}, NonzeroPadFillCase{"fill_1", 1.0f}),
                         FuseAvgDownRejectsNonzeroPadFillTests::getTestCaseName);

TEST_F(TransformationTestsF, FuseAvgDownRejectsMisplacedDynamicTemporalPadding) {
    AvgDownParams p;
    p.output_channels = 192;
    p.factor_t = 2;
    p.add_pad = true;
    p.dynamic_temporal_pad = true;
    p.padding_variant = DynamicPaddingVariant::MisplacedTemporal;
    model = getModel(p);
    manager.register_pass<ov::intel_gpu::FuseAvgDown>();
}

TEST_F(TransformationTestsF, FuseAvgDownRejectsNonzeroDynamicSpatialPadding) {
    AvgDownParams p;
    p.output_channels = 192;
    p.factor_t = 2;
    p.add_pad = true;
    p.dynamic_temporal_pad = true;
    p.padding_variant = DynamicPaddingVariant::NonzeroSpatial;
    model = getModel(p);
    manager.register_pass<ov::intel_gpu::FuseAvgDown>();
}

TEST_F(TransformationTestsF, FuseAvgDownRejectsUnexpectedGroupSize) {
    AvgDownParams p;
    p.output_channels = 128;
    p.factor_t = 2;
    p.add_pad = true;
    model = getModel(p);
    manager.register_pass<ov::intel_gpu::FuseAvgDown>();
}

// The group axis written as a negative axis is the same reduction
class FuseAvgDownNegativeAxisTests : public TransformationTestsF, public testing::WithParamInterface<std::tuple<int64_t, int64_t, int64_t>> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::tuple<int64_t, int64_t, int64_t>>& obj) {
        const auto& [input_channels, output_channels, factor_t] = obj.param;
        return "C" + std::to_string(input_channels) + "to" + std::to_string(output_channels) + "_ft" + std::to_string(factor_t) + "_fs2_axis_neg4";
    }
};

TEST_P(FuseAvgDownNegativeAxisTests, Rewrites) {
    const auto& [input_channels, output_channels, factor_t] = GetParam();
    AvgDownParams p;
    p.input_channels = input_channels;
    p.output_channels = output_channels;
    p.factor_t = factor_t;
    p.add_pad = factor_t > 1;
    p.reduce_axis = -4;
    model = getModel(p);
    model_ref = getModelRef(p);
    manager.register_pass<ov::intel_gpu::FuseAvgDown>();
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

INSTANTIATE_TEST_SUITE_P(smoke_FuseAvgDown,
                         FuseAvgDownNegativeAxisTests,
                         testing::Values(std::make_tuple(96, 96, 1), std::make_tuple(96, 192, 2)),
                         FuseAvgDownNegativeAxisTests::getTestCaseName);

class FuseAvgDownRejectsOtherAxisTests : public TransformationTestsF, public testing::WithParamInterface<int64_t> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<int64_t>& obj) {
        const int64_t axis = obj.param;
        return "axis_" + (axis < 0 ? "neg" + std::to_string(-axis) : std::to_string(axis));
    }
};

TEST_P(FuseAvgDownRejectsOtherAxisTests, Rejects) {
    AvgDownParams p;
    p.reduce_axis = GetParam();
    model = getModel(p);
    manager.register_pass<ov::intel_gpu::FuseAvgDown>();
}

// -3 and 3 are the T' axis, 1 the channel axis: a mean over them is not an AvgDown
INSTANTIATE_TEST_SUITE_P(smoke_FuseAvgDown, FuseAvgDownRejectsOtherAxisTests, testing::Values(1, 3, -3), FuseAvgDownRejectsOtherAxisTests::getTestCaseName);

TEST_F(TransformationTestsF, FuseAvgDownRejectsUnexpectedTransposeOrder) {
    AvgDownParams p;
    p.transpose_order = {0, 1, 3, 5, 6, 2, 4, 7};
    model = getModel(p);
    manager.register_pass<ov::intel_gpu::FuseAvgDown>();
}

TEST_F(TransformationTestsF, FuseAvgDownRejectsIntermediateWithSeveralConsumers) {
    // The Transpose output is also consumed outside the chain, so the chain would stay alive next to the
    // fused operation and its work would be duplicated
    AvgDownParams p;
    p.fold_reshapes = true;
    model = getModel(p);
    for (const auto& node : model->get_ops()) {
        if (ov::is_type<ov::op::v1::Transpose>(node)) {
            model->add_results({std::make_shared<ov::op::v0::Result>(std::make_shared<ov::op::v0::Relu>(node))});
        }
    }
    manager.register_pass<ov::intel_gpu::FuseAvgDown>();
}

TEST_F(TransformationTestsF, FuseAvgDownRejectsIdentityChainNotSplittingInput) {
    // Unit factors, but the 8-D view is [1, 4, 3, 1, 2, 1, 3, 1] while the input is [1, 4, 2, 3, 3]:
    // the chain reshapes the data to [1, 4, 3, 2, 3], so it must not be replaced by its input
    model = getLiteralChainModel(ov::Shape{1, 4, 2, 3, 3}, {1, 4, 3, 1, 2, 1, 3, 1}, {1, 4, 1, 3, 2, 3});
    manager.register_pass<ov::intel_gpu::FuseAvgDown>();
}

TEST_F(TransformationTestsF, FuseAvgDownRejectsSpatialChainNotSplittingInput) {
    // H and W of the 8-D view are taken from the wrong input axes: [1, 4, 1, 1, 4, 2, 2, 2] for an
    // input [1, 4, 1, 4, 8]; same element count, but not a 2x2 spatial split of the input
    model = getLiteralChainModel(ov::Shape{1, 4, 1, 4, 8}, {1, 4, 1, 1, 4, 2, 2, 2}, {1, 4, 4, 1, 4, 2});
    manager.register_pass<ov::intel_gpu::FuseAvgDown>();
}

TEST(GroupedSpaceToDepthTest, InfersIntervalDimensions) {
    auto parameter =
        std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                ov::PartialShape{ov::Dimension(1, 4), 96, ov::Dimension(3, 8), ov::Dimension(8, 12), ov::Dimension(10, 14)});
    auto grouped_space_to_depth = std::make_shared<ov::intel_gpu::op::GroupedSpaceToDepth>(parameter, 2, 2, 192);

    EXPECT_EQ(grouped_space_to_depth->get_output_partial_shape(0),
              ov::PartialShape({ov::Dimension(1, 4), 192, ov::Dimension(2, 4), ov::Dimension(4, 6), ov::Dimension(5, 7)}));
}

TEST(GroupedSpaceToDepthTest, InfersTemporalCeilDivision) {
    const std::vector<std::pair<ov::Dimension, ov::Dimension>> cases = {
        {ov::Dimension(2), ov::Dimension(1)},
        {ov::Dimension(4), ov::Dimension(2)},
        {ov::Dimension(5), ov::Dimension(3)},
        {ov::Dimension(2, 4), ov::Dimension(1, 2)},
        {ov::Dimension(3, 8), ov::Dimension(2, 4)},
        {ov::Dimension(2, -1), ov::Dimension(1, -1)},
    };
    for (const auto& [time, expected_time] : cases) {
        auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 96, time, 4, 6});
        auto grouped_space_to_depth = std::make_shared<ov::intel_gpu::op::GroupedSpaceToDepth>(parameter, 2, 2, 192);
        EXPECT_EQ(grouped_space_to_depth->get_output_partial_shape(0), ov::PartialShape({1, 192, expected_time, 2, 3})) << "T = " << time;
    }
}

}  // namespace
