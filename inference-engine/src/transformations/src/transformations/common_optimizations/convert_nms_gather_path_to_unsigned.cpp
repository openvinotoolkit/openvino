// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/convert_nms_gather_path_to_unsigned.hpp"
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/op/util/broadcast_base.hpp>
#include <ngraph/op/util/gather_base.hpp>
#include <ngraph/rt_info.hpp>
#include <memory>
#include "itt.hpp"
#include "ngraph/node.hpp"

using namespace ngraph;
using namespace std;

class InitNMSPath: public pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;

    InitNMSPath() {
        MATCHER_SCOPE(InitNMSPath);

        auto nms_pattern = pattern::wrap_type<opset1::NonMaxSuppression,
                opset3::NonMaxSuppression,
                opset5::NonMaxSuppression>();

        matcher_pass_callback callback = [=](pattern::Matcher &m) {
            const auto& out_nodes = m.get_match_root()->output(0).get_target_inputs();
            for (const auto& out_node : out_nodes) {
                auto& out_rt_info = out_node.get_node()->get_rt_info();
                out_rt_info["NMS_SELECTED_INDICES"] = make_shared<VariantWrapper<string>>("");
            }
            return true;
        };

        auto m = make_shared<pattern::Matcher>(nms_pattern, matcher_name);
        register_matcher(m, callback);
    }
};

NGRAPH_RTTI_DEFINITION(InitNMSPath, "InitNMSPath", 0);


class PropagateNMSPath: public pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;

    PropagateNMSPath(){
        MATCHER_SCOPE(PropagateNMSPath);

        auto node_pattern = pattern::wrap_type<
                opset8::Squeeze,
                opset8::Unsqueeze,
                opset8::Reshape,
                op::util::BroadcastBase,
                opset8::StridedSlice,
                opset8::VariadicSplit,
                opset8::Concat,
                opset8::Convert>();

        matcher_pass_callback callback = [=](pattern::Matcher &m) {
            auto node = m.get_match_root();
            const auto & inputs = node->input_values();
            if (any_of(inputs.begin(), inputs.end(), [](const Output<Node> & output) {
                return output.get_node()->get_rt_info().count("NMS_SELECTED_INDICES");
            })) {
                auto & rt_info = node->get_rt_info();
                rt_info["NMS_SELECTED_INDICES"] = make_shared<VariantWrapper<string>>("");
            }
            return true;
        };

        auto m = make_shared<pattern::Matcher>(node_pattern, matcher_name);
        register_matcher(m, callback);
    }
};

NGRAPH_RTTI_DEFINITION(PropagateNMSPath, "PropagateNMSPath", 0);

class UpdateConvertGather: public pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;

    UpdateConvertGather(){
        MATCHER_SCOPE(UpdateConvertGather);

        auto node_pattern = pattern::wrap_type<op::util::GatherBase>();

        matcher_pass_callback callback = [=](pattern::Matcher &m) {
            auto gather = m.get_match_root();
            auto indices = gather->input_value(1);

            const auto& rt_info = indices.get_node()->get_rt_info();
            if (!rt_info.count("NMS_SELECTED_INDICES"))
                return false;

            auto out_type = (indices.get_element_type() == element::i64 ?  element::u64 : element::u32);
            auto existing_convert = dynamic_pointer_cast<opset8::Convert>(indices.get_node_shared_ptr());
            if (existing_convert && indices.get_target_inputs().size() == 1) {
                existing_convert->set_convert_element_type(out_type);
                existing_convert->validate_and_infer_types();
            } else {
                auto new_convert_to_unsigned = make_shared<opset8::Convert>(indices, out_type);
                gather->input(1).replace_source_output(new_convert_to_unsigned);
                copy_runtime_info(gather, new_convert_to_unsigned);
            }
            return true;
        };

        auto m = make_shared<pattern::Matcher>(node_pattern, matcher_name);
        register_matcher(m, callback);
    }
};

NGRAPH_RTTI_DEFINITION(UpdateConvertGather, "UpdateConvertGather", 0);

pass::ConvertNmsGatherPathToUnsigned::ConvertNmsGatherPathToUnsigned() {
    add_matcher<InitNMSPath>();
    add_matcher<PropagateNMSPath>();
    add_matcher<UpdateConvertGather>();
}

NGRAPH_RTTI_DEFINITION(pass::ConvertNmsGatherPathToUnsigned, "ConvertNmsGatherPathToUnsigned", 0);
