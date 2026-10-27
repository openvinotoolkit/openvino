// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/multi_scale_deformable_attn_grid_sample_fusion.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/grid_sample.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/msda.hpp"

namespace {

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v8 = ov::op::v8;
namespace v9 = ov::op::v9;
using namespace ov::pass::pattern;

// One feature level: GridSample of VariadicSplit output l of value, sampled at
// the coordinates 2 * locations[:, :, :, l] - 1.
struct LevelPattern {
    std::shared_ptr<ov::Node> value, image_flat, locations, gather, root;

    LevelPattern() {
        value = any_input(has_static_shape() && shape_matches("B, S, H, D"));
        auto split = wrap_type<v1::VariadicSplit>({value, 1, any_input()});
        image_flat = wrap_type<v1::Reshape>({split, any_input()}, shape_matches("B, HW, HD"));
        auto image_transpose = wrap_type<v1::Transpose>({image_flat, {0, 2, 1}});
        auto image = wrap_type<v1::Reshape>({image_transpose, any_input()}, shape_matches("BH, D, h, w"));

        locations = any_input(has_static_shape() && shape_matches("B, Q, H, L, P, 2"));
        auto two = ov::pass::pattern::optional<v0::Convert>(wrap_type<v0::Constant>(value_matches("2")));
        auto minus_one = ov::pass::pattern::optional<v0::Convert>(wrap_type<v0::Constant>(value_matches("-1")));
        auto coords = wrap_type<v1::Add>({wrap_type<v1::Multiply>({locations, two}), minus_one});
        gather = wrap_type<v8::Gather>(
            {coords, wrap_const(), wrap_type<v0::Constant>(value_matches("3") || value_matches("-3"))},
            {{"batch_dims", 0}});
        // A [1] shaped Gather index keeps the level axis until a Squeeze (or an
        // equivalent Reshape) removes it.
        auto level_coords = ov::pass::pattern::optional<v0::Squeeze, v1::Reshape>({gather, any_input()});
        auto coords_transpose =
            wrap_type<v1::Transpose>({level_coords, {0, 2, 1, 3, 4}}, shape_matches("B, H, Q, P, 2"));
        auto grid_coords = wrap_type<v1::Reshape>({coords_transpose, any_input()}, shape_matches("BH, Q, P, 2"));
        // MSDA takes value and locations of one element type.
        auto grid = wrap_type<v9::GridSample>(
            {image, grid_coords},
            attrs_match({{"align_corners", false}, {"mode", "bilinear"}, {"padding_mode", "zeros"}}) &&
                ov::pass::pattern::op::Predicate(
                    [](const ov::Output<ov::Node>& output) {
                        const auto node = output.get_node();
                        return node->get_input_element_type(0) == node->get_input_element_type(1);
                    },
                    "same_data_and_grid_type"));
        root = wrap_type<v0::Unsqueeze, v1::Reshape>({grid, any_input()}, shape_matches("BH, D, Q, 1, P"));
    }
};

struct Level {
    const ov::Node* split;
    ov::Output<ov::Node> value, locations;
    int64_t h, w;
    ov::NodeVector nodes;
};

bool same_value(const PatternSymbolMap& symbols, const std::string& name, const PatternSymbolValue& value) {
    const auto it = symbols.find(name);
    return it == symbols.end() || it->second == value;
}

// Matches every input of the Concat against the level pattern. The checks
// below relate several levels or multiply dimensions, which the shape notation
// cannot express: level l reads output l of one VariadicSplit and
// locations[:, :, :, l], all levels share the value and locations tensors, the
// level sizes add up to the S keys, BH = B * H and HD = H * D. The dimension
// names shared with the aggregation pattern are added to `symbols`, so that
// pattern checks them as well; h, w and HW are bound per level only.
std::optional<std::vector<Level>> match_levels(const LevelPattern& pattern,
                                               const ov::Output<ov::Node>& concat_output,
                                               PatternSymbolMap& symbols) {
    const auto concat = ov::as_type_ptr<v0::Concat>(concat_output.get_node_shared_ptr());
    // The levels are concatenated along axis 3 (-2) of the [B*H, D, Q, 1, P] samples.
    if (!concat || concat->get_output_partial_shape(0).rank() != 5 ||
        (concat->get_axis() < 0 ? concat->get_axis() + 5 : concat->get_axis()) != 3)
        return std::nullopt;

    std::vector<Level> levels;
    PatternSymbolMap level_dims;
    int64_t keys = 0;
    for (size_t l = 0; l < concat->get_input_size(); ++l) {
        Matcher matcher(pattern.root, "MultiScaleDeformableAttnGridSampleFusionLevel");
        if (!matcher.match(concat->input_value(l)))
            return std::nullopt;
        const auto& pm = matcher.get_pattern_value_map();
        const auto& dims = matcher.get_symbols();
        const auto split_output = pm.at(pattern.image_flat).get_node()->input_value(0);
        Level level{split_output.get_node(),
                    pm.at(pattern.value),
                    pm.at(pattern.locations),
                    dims.at("h").i(),
                    dims.at("w").i(),
                    matcher.get_matched_nodes()};
        const auto index =
            ov::as_type_ptr<v0::Constant>(pm.at(pattern.gather).get_node()->get_input_node_shared_ptr(1));
        if (!index || split_output.get_index() != l ||
            index->cast_vector<int64_t>() != std::vector<int64_t>{static_cast<int64_t>(l)} ||
            dims.at("HW").i() != level.h * level.w ||
            (!levels.empty() && (level.split != levels.front().split || level.value != levels.front().value ||
                                 level.locations != levels.front().locations)))
            return std::nullopt;
        for (const char* name : {"B", "S", "H", "D", "Q", "L", "P", "BH", "HD"}) {
            if (!same_value(level_dims, name, dims.at(name)) || !same_value(symbols, name, dims.at(name)))
                return std::nullopt;
            level_dims.emplace(name, dims.at(name));
        }
        keys += level.h * level.w;
        levels.push_back(std::move(level));
    }
    const auto dim = [&level_dims](const std::string& name) {
        return level_dims.at(name).i();
    };
    if (levels.empty() || dim("L") != static_cast<int64_t>(levels.size()) || dim("S") != keys ||
        dim("BH") != dim("B") * dim("H") || dim("HD") != dim("H") * dim("D"))
        return std::nullopt;
    symbols.insert(level_dims.begin(), level_dims.end());
    return levels;
}

std::optional<ov::Shape> static_shape(const ov::Output<ov::Node>& output, size_t rank) {
    const auto& shape = output.get_partial_shape();
    if (shape.is_dynamic() || shape.size() != rank)
        return std::nullopt;
    return shape.to_shape();
}

std::vector<int64_t> constant_values(const ov::Output<ov::Node>& output) {
    const auto constant = ov::as_type_ptr<v0::Constant>(output.get_node_shared_ptr());
    return constant ? constant->cast_vector<int64_t>() : std::vector<int64_t>{};
}

bool all_equal(const std::vector<int64_t>& values, int64_t expected) {
    return !values.empty() && std::all_of(values.begin(), values.end(), [expected](int64_t v) {
        return v == expected;
    });
}

// The value of a constant whose elements are all equal, possibly behind a decompression Convert.
std::optional<double> scalar_constant(const ov::Output<ov::Node>& output) {
    auto node = output.get_node_shared_ptr();
    if (ov::is_type<v0::Convert>(node))
        node = node->get_input_node_shared_ptr(0);
    const auto constant = ov::as_type_ptr<v0::Constant>(node);
    if (!constant)
        return std::nullopt;
    const auto values = constant->cast_vector<double>();
    if (values.empty() || std::any_of(values.begin(), values.end(), [&values](double v) {
            return v != values.front();
        }))
        return std::nullopt;
    return values.front();
}

bool is_filled_constant(const ov::Output<ov::Node>& output, float expected) {
    const auto value = scalar_constant(output);
    return value && *value == static_cast<double>(expected);
}

// Reshape, Squeeze and Unsqueeze keep the row-major element order, so a chain of them is skipped as a whole.
ov::Output<ov::Node> skip_reshapes(ov::Output<ov::Node> output, ov::NodeVector& visited) {
    while (ov::is_type<v1::Reshape>(output.get_node()) || ov::is_type<v0::Squeeze>(output.get_node()) ||
           ov::is_type<v0::Unsqueeze>(output.get_node())) {
        visited.push_back(output.get_node_shared_ptr());
        output = output.get_node()->input_value(0);
    }
    return output;
}

std::optional<ov::Output<ov::Node>> transpose_input(const ov::Output<ov::Node>& output,
                                                    const std::vector<int64_t>& order,
                                                    ov::NodeVector& visited) {
    const auto transpose = ov::as_type_ptr<v1::Transpose>(output.get_node_shared_ptr());
    if (!transpose || constant_values(transpose->input_value(1)) != order)
        return std::nullopt;
    visited.push_back(transpose);
    return transpose->input_value(0);
}

// A Slice that keeps every element (start 0, step 1, unchanged shape) is a no-op.
bool is_identity_slice(const std::shared_ptr<ov::Node>& node) {
    const auto slice = ov::as_type_ptr<v8::Slice>(node);
    return slice && slice->get_input_partial_shape(0) == slice->get_output_partial_shape(0) &&
           all_equal(constant_values(slice->input_value(1)), 0) && all_equal(constant_values(slice->input_value(3)), 1);
}

// GridSample coordinates expressed as scale * Transpose(0,2,1,3,4)(source) + shift.
struct GridCoords {
    ov::Output<ov::Node> source;
    double scale = 1.0;
    double shift = 0.0;
};

// Walks back from the GridSample coordinates through reshapes, Converts, a no-op Slice, the Transpose and scalar
// Multiply/Add/Subtract: GPU low-precision passes rewrite and move the 2 * x - 1 normalization across the Transpose.
std::optional<GridCoords> grid_coords_source(ov::Output<ov::Node> output, ov::NodeVector& visited) {
    GridCoords coords;
    bool transposed = false;
    while (true) {
        const auto node = output.get_node_shared_ptr();
        if (ov::is_type<v1::Reshape>(node) || ov::is_type<v0::Squeeze>(node) || ov::is_type<v0::Unsqueeze>(node) ||
            ov::is_type<v0::Convert>(node) || is_identity_slice(node)) {
            visited.push_back(node);
            output = node->input_value(0);
            continue;
        }
        if (const auto transpose = ov::as_type_ptr<v1::Transpose>(node)) {
            if (transposed || constant_values(transpose->input_value(1)) != std::vector<int64_t>{0, 2, 1, 3, 4})
                break;
            transposed = true;
            visited.push_back(node);
            output = node->input_value(0);
            continue;
        }
        if (!ov::is_type<v1::Multiply>(node) && !ov::is_type<v1::Add>(node) && !ov::is_type<v1::Subtract>(node))
            break;
        std::optional<size_t> data_index;
        std::optional<double> k;
        for (size_t i = 0; i < 2 && !data_index; ++i) {
            k = scalar_constant(node->input_value(1 - i));
            if (k)
                data_index = i;
        }
        if (!data_index || node->get_output_partial_shape(0) != node->get_input_partial_shape(*data_index))
            break;
        if (ov::is_type<v1::Multiply>(node)) {
            coords.scale *= *k;
        } else if (ov::is_type<v1::Add>(node)) {
            coords.shift += coords.scale * *k;
        } else if (*data_index == 0) {
            coords.shift -= coords.scale * *k;
        } else {
            coords.shift += coords.scale * *k;
            coords.scale = -coords.scale;
        }
        visited.push_back(node);
        output = node->input_value(*data_index);
    }
    if (!transposed)
        return std::nullopt;
    coords.source = output;
    return coords;
}

}  // namespace

ov::pass::MultiScaleDeformableAttnGridSampleFusion::MultiScaleDeformableAttnGridSampleFusion() {
    MATCHER_SCOPE(MultiScaleDeformableAttnGridSampleFusion);

    const auto level_pattern = std::make_shared<LevelPattern>();
    auto concat_m =
        wrap_type<v0::Concat>(shape_matches("BH, D, Q, L, P") &&
                              ov::pass::pattern::op::Predicate(
                                  [level_pattern](PatternSymbolMap& symbols, const ov::Output<ov::Node>& output) {
                                      return match_levels(*level_pattern, output, symbols).has_value();
                                  },
                                  "msda_levels_match"));
    auto values_m = wrap_type<v1::Reshape>({concat_m, any_input()}, shape_matches("BH, D, Q, LP"));
    auto weights_m = any_input(has_static_shape() && shape_matches("B, Q, H, L, P"));
    auto weights_transpose_m = wrap_type<v1::Transpose>({weights_m, {0, 2, 1, 3, 4}});
    auto weights_reshape_m = wrap_type<v1::Reshape>({weights_transpose_m, any_input()}, shape_matches("BH, 1, Q, LP"));
    auto weighted_m = wrap_type<v1::Multiply>({values_m, weights_reshape_m});
    auto reduce_m = wrap_type<v1::ReduceSum>({weighted_m, -1}, {{"keep_dims", false}});
    // On GPUs without XMX, ConvertReduceToPooling has already turned an f16
    // ReduceSum into AvgPool over the last axis times the number of samples.
    auto pool_m = wrap_type<v1::AvgPool>({weighted_m},
                                         attrs_match({{"strides", std::vector<int64_t>{1, 1}},
                                                      {"pads_begin", std::vector<int64_t>{0, 0}},
                                                      {"pads_end", std::vector<int64_t>{0, 0}}}) &&
                                             shape_matches("BH, D, Q, 1"));
    auto pool_sum_m = wrap_type<v1::Multiply>({pool_m, wrap_type<v0::Constant>(value_matches("LP"))});
    auto sum_m = reduce_m | ov::pass::pattern::optional<v1::Reshape>({pool_sum_m, any_input()});
    auto heads_m = wrap_type<v1::Reshape>({sum_m, any_input()}, shape_matches("B, HD, Q"));
    // When the subgraph is a model output that f16 inference keeps in f32, the
    // GPU pipeline reaches this pass with a Convert before the last Transpose.
    auto heads_convert_m = ov::pass::pattern::optional<v0::Convert>({heads_m});
    auto output_m = wrap_type<v1::Transpose>({heads_convert_m, {0, 2, 1}});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto output = pattern_map.at(output_m).get_node_shared_ptr();
        if (transformation_callback(output))
            return false;

        auto symbols = m.get_symbols();
        const auto levels = match_levels(*level_pattern, pattern_map.at(concat_m), symbols);
        if (!levels)
            return false;

        std::vector<int32_t> spatial_shapes, level_starts;
        int64_t start = 0;
        ov::NodeVector fused = m.get_matched_nodes();
        for (const auto& level : *levels) {
            spatial_shapes.push_back(static_cast<int32_t>(level.h));
            spatial_shapes.push_back(static_cast<int32_t>(level.w));
            level_starts.push_back(static_cast<int32_t>(start));
            start += level.h * level.w;
            fused.insert(fused.end(), level.nodes.begin(), level.nodes.end());
        }
        const auto num_levels = levels->size();
        auto shapes = v0::Constant::create(ov::element::i32, ov::Shape{num_levels, 2}, spatial_shapes);
        auto starts = v0::Constant::create(ov::element::i32, ov::Shape{num_levels}, level_starts);
        auto msda = std::make_shared<ov::op::internal::MSDA>(levels->front().value,
                                                             shapes,
                                                             starts,
                                                             levels->front().locations,
                                                             pattern_map.at(weights_m));
        ov::NodeVector created{shapes, starts, msda};
        std::shared_ptr<ov::Node> result = msda;
        if (pattern_map.count(heads_convert_m)) {
            result = std::make_shared<v0::Convert>(msda, output->get_output_element_type(0));
            created.push_back(result);
        }
        result->set_friendly_name(output->get_friendly_name());
        ov::copy_runtime_info(fused, created);
        ov::replace_node(output, result);
        return true;
    };

    auto m = std::make_shared<Matcher>(output_m, matcher_name);
    register_matcher(m, callback);
}

ov::pass::SingleScaleDeformableAttnGridSampleFusion::SingleScaleDeformableAttnGridSampleFusion() {
    MATCHER_SCOPE(SingleScaleDeformableAttnGridSampleFusion);

    auto grid_m = wrap_type<v9::GridSample>(
        {any_input(), any_input()},
        attrs_match({{"align_corners", false}, {"mode", "bilinear"}, {"padding_mode", "zeros"}}));
    auto weighted_m = wrap_type<v1::Multiply>({grid_m, any_input()});
    auto reduce_m = wrap_type<v1::ReduceSum>({weighted_m, wrap_const()}, {{"keep_dims", false}});
    // Same ConvertReduceToPooling form as in MultiScaleDeformableAttnGridSampleFusion.
    auto pool_m = wrap_type<v1::AvgPool>({weighted_m},
                                         attrs_match({{"strides", std::vector<int64_t>{1, 1}},
                                                      {"pads_begin", std::vector<int64_t>{0, 0}},
                                                      {"pads_end", std::vector<int64_t>{0, 0}}}));
    auto pool_sum_m = wrap_type<v1::Multiply>({pool_m, wrap_const()});
    auto sum_m = reduce_m | ov::pass::pattern::optional<v1::Reshape>({pool_sum_m, any_input()});
    auto output_m = wrap_type<v1::Reshape>({sum_m, any_input()});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto output = pattern_map.at(output_m).get_node_shared_ptr();
        if (transformation_callback(output))
            return false;
        const auto grid = pattern_map.at(grid_m).get_node_shared_ptr();
        const auto weighted = pattern_map.at(weighted_m).get_node_shared_ptr();
        ov::NodeVector fused = m.get_matched_nodes();

        // GridSample reads a [B*H, D, h, w] image at [B*H, Q, P, 2] coordinates.
        const auto image_shape = static_shape(grid->input_value(0), 4);
        const auto coords_shape = static_shape(grid->input_value(1), 4);
        const auto output_shape = static_shape(output->output(0), 3);
        if (!image_shape || !coords_shape || !output_shape || (*coords_shape)[3] != 2 ||
            (*coords_shape)[0] != (*image_shape)[0])
            return false;
        const size_t BH = (*image_shape)[0], D = (*image_shape)[1], h = (*image_shape)[2], w = (*image_shape)[3];
        const size_t Q = (*coords_shape)[1], P = (*coords_shape)[2];

        if (pattern_map.count(reduce_m)) {
            const auto axes = constant_values(pattern_map.at(reduce_m).get_node()->input_value(1));
            if (axes.size() != 1 || (axes[0] != -1 && axes[0] != 3))
                return false;
        } else {
            const auto pool = ov::as_type_ptr<v1::AvgPool>(pattern_map.at(pool_m).get_node_shared_ptr());
            const auto pool_sum = pattern_map.at(pool_sum_m).get_node();
            if (!pool || pool->get_kernel() != ov::Shape{1, P} ||
                !(is_filled_constant(pool_sum->input_value(0), static_cast<float>(P)) ||
                  is_filled_constant(pool_sum->input_value(1), static_cast<float>(P))))
                return false;
        }

        // value [B, S, H*D] -> Transpose(0,2,1) -> reshapes -> [B*H, D, h, w]
        const auto value = transpose_input(skip_reshapes(grid->input_value(0), fused), {0, 2, 1}, fused);
        const auto value_shape = value ? static_shape(*value, 3) : std::nullopt;
        if (!value_shape)
            return false;
        const size_t B = (*value_shape)[0], S = (*value_shape)[1], HD = (*value_shape)[2];
        if (B == 0 || D == 0 || BH % B != 0)
            return false;
        const size_t H = BH / B;
        if (S != h * w || HD != H * D || *output_shape != ov::Shape{B, HD, Q})
            return false;

        // locations [B, Q, H, P, 2] in [0, 1] reach GridSample as 2 * x - 1, transposed and reshaped to [B*H, Q, P, 2]
        const auto coords = grid_coords_source(grid->input_value(1), fused);
        if (!coords || static_shape(coords->source, 5) != ov::Shape{B, Q, H, P, 2} || coords->scale == 0.0)
            return false;

        // weights [B, Q, H, P] -> Transpose(0,2,1,3) -> [B*H, 1, Q, P]
        const auto weights_input =
            weighted->input_value(0).get_node() == grid.get() ? weighted->input_value(1) : weighted->input_value(0);
        if (static_shape(weights_input, 4) != ov::Shape{BH, 1, Q, P})
            return false;
        const auto weights = transpose_input(skip_reshapes(weights_input, fused), {0, 2, 1, 3}, fused);
        if (!weights || static_shape(*weights, 4) != ov::Shape{B, Q, H, P})
            return false;

        const auto type = value->get_element_type();
        if (!type.is_real())
            return false;
        ov::NodeVector created;
        const auto as_type = [&](ov::Output<ov::Node> input) {
            if (input.get_element_type() == type)
                return input;
            auto convert = std::make_shared<v0::Convert>(input, type);
            created.push_back(convert);
            return convert->output(0);
        };
        const auto i64 = [&created](const std::vector<size_t>& dims) {
            auto constant = v0::Constant::create(ov::element::i64, ov::Shape{dims.size()}, dims);
            created.push_back(constant);
            return constant;
        };
        const auto scalar = [&created, &type](double value) {
            auto constant = v0::Constant::create(type, ov::Shape{}, {value});
            created.push_back(constant);
            return constant;
        };

        // MSDA samples at 2 * x - 1, so x = (scale * source + shift + 1) / 2.
        auto locations = as_type(coords->source);
        const double scale = coords->scale / 2.0, shift = (coords->shift + 1.0) / 2.0;
        if (scale != 1.0) {
            auto mul = std::make_shared<v1::Multiply>(locations, scalar(scale));
            created.push_back(mul);
            locations = mul->output(0);
        }
        if (shift != 0.0) {
            auto add = std::make_shared<v1::Add>(locations, scalar(shift));
            created.push_back(add);
            locations = add->output(0);
        }
        auto value_4d = std::make_shared<v1::Reshape>(*value, i64({B, S, H, D}), false);
        auto locations_6d = std::make_shared<v1::Reshape>(locations, i64({B, Q, H, 1, P, 2}), false);
        auto weights_5d = std::make_shared<v1::Reshape>(as_type(*weights), i64({B, Q, H, 1, P}), false);
        auto shapes = v0::Constant::create(ov::element::i32,
                                           ov::Shape{1, 2},
                                           std::vector<int32_t>{static_cast<int32_t>(h), static_cast<int32_t>(w)});
        auto starts = v0::Constant::create(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{0});
        auto msda = std::make_shared<ov::op::internal::MSDA>(value_4d, shapes, starts, locations_6d, weights_5d);
        std::shared_ptr<ov::Node> result = std::make_shared<v1::Transpose>(msda, i64({0, 2, 1}));
        created.insert(created.end(), {value_4d, locations_6d, weights_5d, shapes, starts, msda, result});
        if (result->get_output_element_type(0) != output->get_output_element_type(0)) {
            result = std::make_shared<v0::Convert>(result, output->get_output_element_type(0));
            created.push_back(result);
        }
        result->set_friendly_name(output->get_friendly_name());
        ov::copy_runtime_info(fused, created);
        ov::replace_node(output, result);
        return true;
    };

    auto m = std::make_shared<Matcher>(output_m, matcher_name);
    register_matcher(m, callback);
}
