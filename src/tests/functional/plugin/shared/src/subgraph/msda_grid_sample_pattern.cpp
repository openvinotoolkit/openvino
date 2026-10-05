// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "shared_test_classes/subgraph/msda_grid_sample_pattern.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/grid_sample.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"

namespace ov {
namespace test {

namespace {

// Shape constants: element type and length follow the vector.
std::shared_ptr<ov::op::v0::Constant> i64_const(std::vector<int64_t> v) {
    return ov::op::v0::Constant::create(element::i64, ov::Shape{v.size()}, v);
}

constexpr size_t kLevels = 3;
constexpr size_t kPoints = 2;
constexpr size_t kHeads = 2;
constexpr size_t kEmbed = 8;
constexpr size_t kQueries = 6;
constexpr size_t kBatch = 1;
const size_t kSpatial[kLevels][2] = {{4, 5}, {6, 6}, {8, 7}};
}  // namespace

void MSDAGridSamplePattern::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();
    for (size_t i = 0; i < funcInputs.size(); ++i) {
        const auto& param = funcInputs[i];
        ov::Tensor tensor(param.get_element_type(), targetInputStaticShapes[i]);
        auto* data = tensor.data<float>();
        const size_t total = tensor.get_size();
        // Locations stay inside (0, 1) so the fused kernel samples the map
        // interior; other inputs cover a wider range.
        const float lo = (i == 1) ? 0.05f : -1.f;
        const float hi = (i == 1) ? 0.95f : 1.f;
        for (size_t j = 0; j < total; ++j) {
            data[j] = lo + (hi - lo) * (static_cast<float>((j * 17 + i * 31) % 101) / 100.f);
        }
        inputs[param.get_node_shared_ptr()] = tensor;
    }
}

void MSDAGridSamplePattern::SetUp() {
    size_t keys = 0;
    for (auto& hw : kSpatial)
        keys += hw[0] * hw[1];

    auto value = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{kBatch, keys, kHeads, kEmbed});
    value->set_friendly_name("value");
    auto loc01 =
        std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{kBatch, kQueries, kHeads, kLevels, kPoints, 2});
    loc01->set_friendly_name("loc01");
    auto weights =
        std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{kBatch, kQueries, kHeads, kLevels, kPoints});
    weights->set_friendly_name("weights");

    auto two = ov::op::v0::Constant::create(element::f32, Shape{1}, {2.f});
    auto minus_one = ov::op::v0::Constant::create(element::f32, Shape{1}, {-1.f});
    auto normalized = std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v1::Multiply>(loc01, two), minus_one);

    std::vector<int64_t> split_sizes;
    for (auto& hw : kSpatial)
        split_sizes.push_back(static_cast<int64_t>(hw[0] * hw[1]));
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(
        value,
        ov::op::v0::Constant::create(element::i64, Shape{}, {1}),
        ov::op::v0::Constant::create(element::i64, Shape{kLevels}, split_sizes));

    OutputVector level_outputs;
    for (size_t l = 0; l < kLevels; ++l) {
        const size_t h = kSpatial[l][0], w = kSpatial[l][1], s = h * w;
        auto image_flat =
            std::make_shared<ov::op::v1::Reshape>(split->output(l),
                                                  i64_const({int64_t(kBatch), int64_t(s), int64_t(kHeads * kEmbed)}),
                                                  true);
        auto image_transpose = std::make_shared<ov::op::v1::Transpose>(image_flat, i64_const({0, 2, 1}));
        auto image = std::make_shared<ov::op::v1::Reshape>(
            image_transpose,
            i64_const({int64_t(kBatch * kHeads), int64_t(kEmbed), int64_t(h), int64_t(w)}),
            true);
        auto gathered =
            std::make_shared<ov::op::v8::Gather>(normalized,
                                                 ov::op::v0::Constant::create(element::i64, Shape{}, {int64_t(l)}),
                                                 ov::op::v0::Constant::create(element::i64, Shape{}, {3}));
        auto coords_transpose = std::make_shared<ov::op::v1::Transpose>(gathered, i64_const({0, 2, 1, 3, 4}));
        auto coords = std::make_shared<ov::op::v1::Reshape>(
            coords_transpose,
            i64_const({int64_t(kBatch * kHeads), int64_t(kQueries), int64_t(kPoints), 2}),
            true);
        ov::op::v9::GridSample::Attributes attributes{false,
                                                      ov::op::v9::GridSample::InterpolationMode::BILINEAR,
                                                      ov::op::v9::GridSample::PaddingMode::ZEROS};
        auto grid = std::make_shared<ov::op::v9::GridSample>(image, coords, attributes);
        level_outputs.push_back(std::make_shared<ov::op::v1::Reshape>(
            grid,
            i64_const({int64_t(kBatch * kHeads), int64_t(kEmbed), int64_t(kQueries), 1, int64_t(kPoints)}),
            true));
    }

    auto concat = std::make_shared<ov::op::v0::Concat>(level_outputs, -2);
    auto values_reshape =
        std::make_shared<ov::op::v1::Reshape>(concat,
                                              i64_const({0, int64_t(kEmbed), 0, int64_t(kLevels * kPoints)}),
                                              true);

    auto weights_value = std::make_shared<ov::op::v1::Reshape>(
        weights,
        i64_const({int64_t(kBatch), int64_t(kQueries), int64_t(kHeads), int64_t(kLevels), int64_t(kPoints)}),
        true);
    auto weights_transpose = std::make_shared<ov::op::v1::Transpose>(weights_value, i64_const({0, 2, 1, 3, 4}));
    auto weights_reshape = std::make_shared<ov::op::v1::Reshape>(
        weights_transpose,
        i64_const({int64_t(kBatch * kHeads), 1, int64_t(kQueries), int64_t(kLevels * kPoints)}),
        true);

    auto mul = std::make_shared<ov::op::v1::Multiply>(values_reshape, weights_reshape);
    auto reduce = std::make_shared<ov::op::v1::ReduceSum>(mul, i64_const({-1}), false);
    auto root = std::make_shared<ov::op::v1::Reshape>(reduce, i64_const({-1, int64_t(kHeads * kEmbed), 0}), true);
    auto output = std::make_shared<ov::op::v1::Transpose>(root, i64_const({0, 2, 1}));

    function = std::make_shared<ov::Model>(ov::OutputVector{output},
                                           ov::ParameterVector{value, loc01, weights},
                                           "MSDAGridSamplePattern");
    targetDevice = ov::test::utils::DEVICE_GPU;
    // The pattern reaches the fusion pass unmodified when the pipeline does not
    // insert FP16 conversions around the f32 subgraph.
    configuration[ov::hint::inference_precision.name()] = ov::element::f32;
    std::vector<InputShape> input_shapes;
    for (const auto& s : {Shape{kBatch, keys, kHeads, kEmbed},
                          Shape{kBatch, kQueries, kHeads, kLevels, kPoints, 2},
                          Shape{kBatch, kQueries, kHeads, kLevels, kPoints}}) {
        input_shapes.push_back({ov::PartialShape{s}, {s}});
    }
    init_input_shapes(input_shapes);
}

}  // namespace test
}  // namespace ov
