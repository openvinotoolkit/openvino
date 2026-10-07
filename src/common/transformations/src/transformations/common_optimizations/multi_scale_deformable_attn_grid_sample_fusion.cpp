// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/multi_scale_deformable_attn_grid_sample_fusion.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/grid_sample.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
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

// Dimension names bound by both the level and the aggregation patterns. The
// level patterns additionally bind h, w and HW per level.
const std::vector<std::string> shared_dims{"B", "S", "H", "D", "Q", "L", "P", "BH", "HD"};

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
        auto two = optional<v0::Convert>(wrap_type<v0::Constant>(value_matches("2")));
        auto minus_one = optional<v0::Convert>(wrap_type<v0::Constant>(value_matches("-1")));
        auto coords = wrap_type<v1::Add>({wrap_type<v1::Multiply>({locations, two}), minus_one});
        gather = wrap_type<v8::Gather>(
            {coords, wrap_const(), wrap_type<v0::Constant>(value_matches("3") || value_matches("-3"))},
            {{"batch_dims", 0}});
        // A [1] shaped Gather index keeps the level axis until a Squeeze removes it.
        auto level_coords = optional<v0::Squeeze, v1::Reshape>({gather, any_input()});
        auto coords_transpose =
            wrap_type<v1::Transpose>({level_coords, {0, 2, 1, 3, 4}}, shape_matches("B, H, Q, P, 2"));
        auto grid_coords = wrap_type<v1::Reshape>({coords_transpose, any_input()}, shape_matches("BH, Q, P, 2"));
        auto grid =
            wrap_type<v9::GridSample>({image, grid_coords},
                                      {{"align_corners", false}, {"mode", "bilinear"}, {"padding_mode", "zeros"}});
        root = wrap_type<v0::Unsqueeze, v1::Reshape>({grid, any_input()}, shape_matches("BH, D, Q, 1, P"));
    }
};

struct Level {
    ov::Output<ov::Node> value, locations;
    int64_t h, w;
    ov::NodeVector nodes;
};

bool same_value(const PatternSymbolMap& symbols, const std::string& name, const PatternSymbolValue& value) {
    const auto it = symbols.find(name);
    return it == symbols.end() || it->second == value;
}

// Matches every input of the Concat against the level pattern. The checks
// relate several levels or are products of dimensions, so the shape notation
// cannot express them: level l reads VariadicSplit output l and
// locations[:, :, :, l], all levels share the value and locations tensors, the
// level sizes add up to the S keys, BH = B * H and HD = H * D. The shared
// dimension names are added to `symbols` so the aggregation pattern checks
// them as well.
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
        Level level{pm.at(pattern.value),
                    pm.at(pattern.locations),
                    dims.at("h").i(),
                    dims.at("w").i(),
                    matcher.get_matched_nodes()};
        const auto split_output = pm.at(pattern.image_flat).get_node()->input_value(0);
        const auto index =
            ov::as_type_ptr<v0::Constant>(pm.at(pattern.gather).get_node()->get_input_node_shared_ptr(1));
        OPENVINO_ASSERT(index,
                        "MultiScaleDeformableAttnGridSampleFusion: the Gather index is expected to be a Constant");
        if (split_output.get_index() != l ||
            index->cast_vector<int64_t>() != std::vector<int64_t>{static_cast<int64_t>(l)} ||
            dims.at("HW").i() != level.h * level.w ||
            (!levels.empty() && (level.value != levels.front().value || level.locations != levels.front().locations)))
            return std::nullopt;
        for (const auto& name : shared_dims) {
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

}  // namespace

ov::pass::MultiScaleDeformableAttnGridSampleFusion::MultiScaleDeformableAttnGridSampleFusion() {
    MATCHER_SCOPE(MultiScaleDeformableAttnGridSampleFusion);
    using ov::pass::pattern::Matcher;
    using ov::pass::pattern::PatternSymbolMap;

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
    auto heads_m = wrap_type<v1::Reshape>({reduce_m, any_input()}, consumers_count(1) && shape_matches("B, HD, Q"));
    auto output_m = wrap_type<v1::Transpose>({heads_m, {0, 2, 1}});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto output = pattern_map.at(output_m).get_node_shared_ptr();
        if (transformation_callback(output))
            return false;

        auto symbols = m.get_symbols();
        const auto levels = match_levels(*level_pattern, pattern_map.at(concat_m), symbols);
        OPENVINO_ASSERT(levels, "MultiScaleDeformableAttnGridSampleFusion: the matched levels are expected to match");

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
        auto msda = std::make_shared<ov::op::internal::MSDA>(ov::OutputVector{levels->front().value,
                                                                              shapes,
                                                                              starts,
                                                                              levels->front().locations,
                                                                              pattern_map.at(weights_m)});
        msda->set_friendly_name(output->get_friendly_name());
        ov::copy_runtime_info(fused, {shapes, starts, msda});
        ov::replace_node(output, msda);
        return true;
    };

    auto m = std::make_shared<Matcher>(output_m, matcher_name);
    register_matcher(m, callback);
}
