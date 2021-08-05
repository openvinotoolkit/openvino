// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/convert_nms_gather_path_to_unsigned.hpp"
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/op/util/broadcast_base.hpp>
#include <ngraph/op/util/gather_base.hpp>
#include <ngraph/rt_info.hpp>
#include <memory>
#include "itt.hpp"
#include "ngraph/node.hpp"

using namespace ngraph;
using namespace std;

class InitNMSPath: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;

    InitNMSPath() {
        MATCHER_SCOPE(InitNMSPath);

        auto nms_pattern = pattern::wrap_type<opset1::NonMaxSuppression,
                opset3::NonMaxSuppression,
                opset5::NonMaxSuppression>();

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
            auto nms_node = dynamic_pointer_cast<Node>(m.get_match_root());
            if (!nms_node)
                return false;

            bool res = false;
            const auto& out_nodes = nms_node->output(0).get_target_inputs();
            for (const auto& out_node : out_nodes) {
                auto& out_rt_info = out_node.get_node()->get_rt_info();
                out_rt_info["NMS_SELECTED_INDICES"] = std::make_shared<ngraph::VariantWrapper<string>>("");
                res = true;
            }
            return res;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(nms_pattern, matcher_name);
        register_matcher(m, callback);
    }
};

NGRAPH_RTTI_DEFINITION(InitNMSPath, "InitNMSPath", 0);


class PropagateNMSPath: public ngraph::pass::MatcherPass {
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

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
            auto node = m.get_match_root();
            auto& rt_info = node->get_rt_info();
            bool res = false;

            for (const auto& in_node : node->input_values()) {
                if (in_node.get_node()->get_rt_info().count("NMS_SELECTED_INDICES")) {
                    rt_info["NMS_SELECTED_INDICES"] = std::make_shared<ngraph::VariantWrapper<string>>("");
                    res = true;
                    break;
                }
            }

            return res;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(node_pattern, matcher_name);
        register_matcher(m, callback);
    }
};

NGRAPH_RTTI_DEFINITION(PropagateNMSPath, "PropagateNMSPath", 0);

class UpdateConvertGather: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;

    UpdateConvertGather(){
        MATCHER_SCOPE(UpdateConvertGather);

        auto node_pattern = pattern::wrap_type<op::util::GatherBase>();

        ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher &m) {
            auto gather = dynamic_pointer_cast<op::util::GatherBase>(m.get_match_root());
            if (!gather)
                return false;

            auto indices = gather->input_value(1);

            const auto& rt_info = indices.get_node()->get_rt_info();
            if (!rt_info.count("NMS_SELECTED_INDICES"))
                return false;

            auto out_type = element::Type_t::u32;
            if (indices.get_element_type() == element::Type_t::i64)
                out_type = element::Type_t::u64;

            if (auto existing_convert = dynamic_pointer_cast<opset8::Convert>(indices.get_node_shared_ptr())) {
                existing_convert->set_convert_element_type(out_type);
            } else {
                auto new_convert_to_unsigned = make_shared<opset8::Convert>(indices, out_type);
                gather->input(1).replace_source_output(new_convert_to_unsigned);
                copy_runtime_info(gather, new_convert_to_unsigned);
            }
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(node_pattern, matcher_name);
        register_matcher(m, callback);
    }
};

NGRAPH_RTTI_DEFINITION(UpdateConvertGather, "UpdateConvertGather", 0);

bool pass::ConvertNmsGatherPathToUnsigned::run_on_function(std::shared_ptr<ngraph::Function> f) {
    RUN_ON_FUNCTION_SCOPE(ConvertToUnsignedNmsGather);

    ngraph::pass::Manager manager;
    manager.register_pass<InitNMSPath>();
    manager.register_pass<PropagateNMSPath>();
    manager.register_pass<UpdateConvertGather>();
    manager.run_passes(f);
    return true;
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertNmsGatherPathToUnsigned, "ConvertNmsGatherPathToUnsigned", 0);
