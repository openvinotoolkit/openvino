// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/gather_sinking_matmul.hpp"

#include <openvino/cc/ngraph/itt.hpp>
#include <transformations/utils/utils.hpp>
#include <utility>

#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/gather_sinking_attr.hpp"
#include "transformations/utils/gather_sinking_utils.hpp"

using namespace ov;
using namespace ov::opset10;
using namespace ov::pass::pattern;
using namespace ov::op::util;
using namespace gather_sinking;
using namespace ov::intel_gna::pass;
using namespace ov::intel_gna::rt_info;

namespace {
/*
Reverts gather indices in a such way that reverted and initial gather will do nothing if
stays after another.
Works only with positive form (no negative indices).
*/
std::vector<int64_t> ReverseGatherIndexes(const std::vector<int64_t>& indexes) {
    std::vector<int64_t> out(indexes.size());
    for (size_t i = 0; i < indexes.size(); i++) {
        out.at(indexes[i]) = i;
    }
    return out;
}

} // namespace

GatherSinkingMatmulForward::GatherSinkingMatmulForward() {
    MATCHER_SCOPE(GatherSinkingMatmulForward);
    auto gather_indices_label = wrap_type<Constant>();
    auto gather_axis_label = wrap_type<Constant>();
    auto gather_label = wrap_type<Gather>({any_input(), gather_indices_label, gather_axis_label});
    auto matmul_label = wrap_type<MatMul>({gather_label, any_input()});

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto gather_indices = as_type_ptr<Constant>(pattern_to_output.at(gather_indices_label).get_node_shared_ptr());
        auto gather_axis = as_type_ptr<Constant>(pattern_to_output.at(gather_axis_label).get_node_shared_ptr());
        auto gather = as_type_ptr<Gather>(pattern_to_output.at(gather_label).get_node_shared_ptr());
        auto matmul = as_type_ptr<MatMul>(pattern_to_output.at(matmul_label).get_node_shared_ptr());

        std::cout << "[EMUTEX DEBUG] GatherSinkingMatmulForward gather " << gather->get_friendly_name() << " matmul " << matmul->get_friendly_name() << std::endl;
        std::cout << "[EMUTEX DEBUG] GatherSinkingMatmulForward gather axis " << gather_axis->cast_vector<int64_t>()[0] << std::endl;

        auto gather_parent = matmul->input_value(0 /* TODO */).get_node()->input_value(0);;
        // insert input gather
#if 0
        size_t gather_axis_value_current = ConvertAxisToPositive(gather_axis->cast_vector<int64_t>()[0],
                                                   gather->get_input_shape(0).size());
#endif
        const size_t gather_axis_value_new = 0; // TODO

        auto gather_axis_new1 = std::make_shared<Constant>(element::i64, Shape{}, gather_axis_value_new);
        auto gather_indices_values = ReverseGatherIndexes(gather_indices->cast_vector<int64_t>());
        auto gather_indices_new1 = std::make_shared<Constant>(element::i64, Shape{gather_indices_values.size()}, gather_indices_values); 
        auto gather_new1 = std::make_shared<Gather>(matmul->input_value(1) /* TODO */, gather_indices_new1, gather_axis_new1);

        matmul->input(1 /* TODO */).replace_source_output(gather_new1->output(0));

        // remove input gather
        matmul->input(0 /* TODO */).replace_source_output(gather_parent);

        // insert output gather
        auto matmul_consumers = matmul->output(0).get_target_inputs();

        auto gather_axis_new2 = gather_axis->clone_with_new_inputs({});
        auto gather_indices_new2 = gather_indices->clone_with_new_inputs({});
        auto gather_new2 = std::make_shared<Gather>(matmul->output(0), gather_indices_new2, gather_axis_new2);

        for (auto& consumer : matmul_consumers) {
            consumer.replace_source_output(gather_new2);
        }

        SwapFriendlyNames(gather_new2, matmul);

        copy_runtime_info(gather, {gather_new1, gather_indices_new1, gather_axis_new1, gather_new2, gather_indices_new2, gather_axis_new2});

        register_new_node(gather_new1);
        register_new_node(gather_new2);

        gather_sinking::UpdateForwardGatherSinkingAbility(gather_new2);

        return true;
    };

    auto m = std::make_shared<Matcher>(matmul_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
