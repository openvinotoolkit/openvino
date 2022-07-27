// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "transformations/common_optimizations/convert_nms_gather_path_to_unsigned.hpp"

#include <memory>
#include <ngraph/op/util/broadcast_base.hpp>
#include <ngraph/op/util/gather_base.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <transformations/rt_info/nms_selected_indices.hpp>

#include "itt.hpp"
#include "ngraph/node.hpp"

using namespace std;

namespace ngraph {
namespace pass {
class InitNMSPath : public pass::MatcherPass {
public:
    OPENVINO_RTTI("InitNMSPath", "0");
    InitNMSPath() {
        MATCHER_SCOPE(InitNMSPath);
        auto nms_pattern =
            pattern::wrap_type<opset1::NonMaxSuppression, opset3::NonMaxSuppression, opset5::NonMaxSuppression>();
        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            const auto& out_nodes = m.get_match_root()->output(0).get_target_inputs();
            for (const auto& out_node : out_nodes) {
                ov::set_nms_selected_indices(out_node.get_node());
            }
            MATCHER_SCOPE_ENABLE(InitNMSPath);
            return true;
        };
        auto m = make_shared<pattern::Matcher>(nms_pattern, matcher_name);
        register_matcher(m, callback);
    }
};
class PropagateNMSPath : public pass::MatcherPass {
public:
    OPENVINO_RTTI("PropagateNMSPath", "0");
    PropagateNMSPath() {
        MATCHER_SCOPE(PropagateNMSPath);
        auto node_pattern = pattern::wrap_type<opset8::Squeeze,
                                               opset8::Unsqueeze,
                                               opset8::Reshape,
                                               op::util::BroadcastBase,
                                               opset8::StridedSlice,
                                               opset8::Slice,
                                               opset8::VariadicSplit,
                                               op::util::GatherBase,
                                               opset8::Concat,
                                               opset8::Convert>();
        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            auto node = m.get_match_root();
            const auto& inputs = node->input_values();
            if (any_of(inputs.begin(), inputs.end(), [](const Output<Node>& output) {
                    MATCHER_SCOPE_ENABLE(PropagateNMSPath);
                    return ov::has_nms_selected_indices(output.get_node());
                })) {
                ov::set_nms_selected_indices(node.get());
            }
            return false;
        };
        auto m = make_shared<pattern::Matcher>(node_pattern, matcher_name);
        register_matcher(m, callback);
    }
};
class UpdateConvertGather : public pass::MatcherPass {
public:
    OPENVINO_RTTI("UpdateConvertGather", "0");
    UpdateConvertGather() {
        MATCHER_SCOPE(UpdateConvertGather);
        auto node_pattern = pattern::wrap_type<op::util::GatherBase>();
        matcher_pass_callback callback = [=](pattern::Matcher& m) {
            auto gather = m.get_match_root();
            auto indices = gather->input_value(1);
            if (!ov::has_nms_selected_indices(indices.get_node()))
                return false;
            gather->get_rt_info()["dontReverseIndices"] = true;
            auto out_type = (indices.get_element_type() == element::i64 ? element::u64 : element::u32);
            auto existing_convert = dynamic_pointer_cast<opset8::Convert>(indices.get_node_shared_ptr());
            if (existing_convert && indices.get_target_inputs().size() == 1) {
                existing_convert->set_convert_element_type(out_type);
                existing_convert->validate_and_infer_types();
            } else {
                auto new_convert_to_unsigned = make_shared<opset8::Convert>(indices, out_type);
                gather->input(1).replace_source_output(new_convert_to_unsigned);
                copy_runtime_info(gather, new_convert_to_unsigned);
            }
            MATCHER_SCOPE_ENABLE(UpdateConvertGather);
            return true;
        };
        auto m = make_shared<pattern::Matcher>(node_pattern, matcher_name);
        register_matcher(m, callback);
    }
};
}  // namespace pass
}  // namespace ngraph

ngraph::pass::ConvertNmsGatherPathToUnsigned::ConvertNmsGatherPathToUnsigned() {
    add_matcher<InitNMSPath>();
    add_matcher<PropagateNMSPath>();
    add_matcher<UpdateConvertGather>();
}
