// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/gather_sinking_fuse.hpp"

#include <openvino/cc/ngraph/itt.hpp>
#include <transformations/utils/utils.hpp>
#include <utility>

#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/gather_sinking_attr.hpp"
#include "transformations/utils/gather_sinking_utils.hpp"

using namespace ov;
using namespace ov::opset9;
using namespace ov::pass::pattern;
using namespace ov::op::util;
using namespace gather_sinking;
using namespace ov::intel_gna::pass;
using namespace ov::intel_gna::rt_info;

namespace {
// TODO: use that function from gather_sinking_utils after merge GatherSinkingBinary
int64_t NormalizeNegativeGatherAxis(int64_t axis, ov::Rank::value_type gather_input_rank) {
    if (axis < 0)
        return axis;
    return axis - gather_input_rank;
}
// TODO: use that function from gather_sinking_utils after merge GatherSinkingBinary
int64_t GetNormalizedNegativeGatherAxis(const std::shared_ptr<Constant>& axis, ov::Rank::value_type gather_input_rank) {
    return NormalizeNegativeGatherAxis(axis->cast_vector<int64_t>()[0], gather_input_rank);
}

// TODO: use constant_has_rank_not_more_than from gather_sinking_utils after merge GatherSinkingBinary
bool IsConstant1D(const Output<Node>& output) {
    return rank_equals(0)(output) || rank_equals(1)(output);
}

int64_t GetGatherAxis(const std::shared_ptr<Gather>& gather) {
    auto output_gather_axis_node = as_type_ptr<Constant>(gather->input_value(2).get_node_shared_ptr());
    return GetNormalizedNegativeGatherAxis(output_gather_axis_node,
                                           gather->get_input_partial_shape(0).rank().get_length());
}

bool IsGatherWithParentGatherSameAxis(const Output<Node>& output) {
    auto output_gather = as_type_ptr<Gather>(output.get_node_shared_ptr());
    if (!output_gather)
        return false;
    const int64_t output_gather_axis = GetGatherAxis(output_gather);
    auto input_gather = as_type_ptr<Gather>(output_gather->input_value(0).get_node_shared_ptr());
    if (!input_gather)
        return false;
    const int64_t input_gather_axis = GetGatherAxis(input_gather);
    return input_gather_axis == output_gather_axis;
}

struct TransformationInfo {
    std::shared_ptr<Constant> input_indices_const;
    std::shared_ptr<Constant> input_axis_const;
    std::shared_ptr<Gather> input_gather;
    std::shared_ptr<Constant> output_indices_const;
    std::shared_ptr<Constant> output_axis_const;
    std::shared_ptr<Gather> output_gather;
};

// TODO: use that function from gather_sinking_utils after merge GatherSinkingBinary
std::vector<int64_t> NormalizeGatherIndices(const std::vector<int64_t>& indices) {
    std::vector<int64_t> normalized(indices.size());
    for (int i = 0; i < indices.size(); ++i) {
        int64_t index = indices[i];
        if (index < 0)
            index += indices.size();
        normalized[i] = index;
    }
    return normalized;
}

/*
Gets gather indices in positive form
*/
// TODO: use that function from gather_sinking_utils after merge GatherSinkingBinary
std::vector<int64_t> GetNormalizedGatherIndices(const std::shared_ptr<Constant>& indices) {
    return NormalizeGatherIndices(indices->cast_vector<int64_t>());
}

std::vector<int64_t> CombineGatherPermutations(const std::vector<int64_t>& input_gather_indices,
                                               const std::vector<int64_t>& output_gather_indices) {
    if (input_gather_indices.size() != output_gather_indices.size())
        return {};
    std::vector<int64_t> result(input_gather_indices.size());
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = input_gather_indices[output_gather_indices[i]];
    }

    return result;
}

bool IsPointlessPermutation(const std::vector<int64_t>& indices) {
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] != i)
            return false;
    }
    return true;
}

std::shared_ptr<Gather> FuseGatherNodes(TransformationInfo& info) {
    const std::vector<int64_t> input_gather_indices = GetNormalizedGatherIndices(info.input_indices_const);
    const std::vector<int64_t> output_gather_indices = GetNormalizedGatherIndices(info.output_indices_const);
    const std::vector<int64_t> result_gather_indices =
        CombineGatherPermutations(input_gather_indices, output_gather_indices);
    if (IsPointlessPermutation(result_gather_indices)) {
        ov::replace_output_update_name(info.output_gather->output(0), info.input_gather->input_value(0));
        return {};
    }

    const auto indices_element_type = info.output_axis_const->get_element_type();
    auto new_indices_const =
        std::make_shared<Constant>(indices_element_type, Shape{result_gather_indices.size()}, result_gather_indices);
    auto new_axis_const = info.output_axis_const->clone_with_new_inputs({});
    auto new_gather = std::make_shared<Gather>(info.input_gather->input_value(0), new_indices_const, new_axis_const);

    ov::replace_node(info.output_gather, new_gather);
    copy_runtime_info(info.input_gather, {new_gather, new_indices_const, new_axis_const});
    new_gather->set_friendly_name(info.output_gather->get_friendly_name());

    return new_gather;
}
}  // namespace

GatherSinkingFuse::GatherSinkingFuse() {
    MATCHER_SCOPE(GatherSinkingFuse);

    auto input_indices_const_label = wrap_type<Constant>(IsConstant1D);
    auto input_axis_const_label = wrap_type<Constant>(IsConstant1D);
    auto input_gather_label = wrap_type<Gather>({any_input(), input_indices_const_label, input_axis_const_label});
    auto output_indices_const_label = wrap_type<Constant>(IsConstant1D);
    auto output_axis_const_label = wrap_type<Constant>(IsConstant1D);
    auto output_gather_label =
        wrap_type<Gather>({input_gather_label, output_indices_const_label, output_axis_const_label},
                          IsGatherWithParentGatherSameAxis);

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        TransformationInfo info;
        info.input_indices_const =
            as_type_ptr<Constant>(pattern_to_output.at(input_indices_const_label).get_node_shared_ptr());
        info.input_axis_const =
            as_type_ptr<Constant>(pattern_to_output.at(input_axis_const_label).get_node_shared_ptr());
        info.input_gather = as_type_ptr<Gather>(pattern_to_output.at(input_gather_label).get_node_shared_ptr());
        info.output_indices_const =
            as_type_ptr<Constant>(pattern_to_output.at(output_indices_const_label).get_node_shared_ptr());
        info.output_axis_const =
            as_type_ptr<Constant>(pattern_to_output.at(output_axis_const_label).get_node_shared_ptr());
        info.output_gather = as_type_ptr<Gather>(pattern_to_output.at(output_gather_label).get_node_shared_ptr());

        auto new_node = FuseGatherNodes(info);
        if (new_node)
            register_new_node(new_node);

        return true;
    };

    auto m = std::make_shared<Matcher>(output_gather_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
