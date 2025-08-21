#include "transformations/common_optimizations/convert_type_relaxed_to_base.hpp"

#include <memory>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/cc/pass/itt.hpp>
#include <openvino/core/node.hpp>
#include <openvino/op/type_relaxed.hpp>
#include <openvino/pass/pattern/matcher.hpp>
#include <transformations/utils/utils.hpp>

ov::pass::ConvertTypeRelaxedToBase::ConvertTypeRelaxedToBase() {
    MATCHER_SCOPE(ConvertTypeRelaxedToBase);
    
    // Match any type_relaxed operation
    auto type_relaxed_op = ngraph::pattern::wrap_type<ov::op::TypeRelaxed<ov::Node>>();
    
    matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto type_relaxed_node = std::dynamic_pointer_cast<ov::op::TypeRelaxed<ov::Node>>(m.get_match_root());
        if (!type_relaxed_node) {
            return false;
        }
        
        // Get the base operation from the type_relaxed wrapper
        auto base_op = type_relaxed_node->get_original_type_info();
        
        // For now, we'll focus on the specific issue mentioned in the bug report
        // which is IsInf operations. We'll convert them to the base opset version
        if (base_op.name == "IsInf") {
            // Create the base IsInf operation
            auto replacement_node = std::make_shared<ov::op::v10::IsInf>(
                type_relaxed_node->input_value(0),
                type_relaxed_node->input_value(1));
            
            // Copy the output names and other attributes
            replacement_node->set_friendly_name(type_relaxed_node->get_friendly_name());
            replacement_node->output(0).get_tensor().set_names(type_relaxed_node->output(0).get_tensor().get_names());
            
            // Replace the node
            ov::replace_node(type_relaxed_node, replacement_node);
            
            // Copy runtime info
            ov::copy_runtime_info(type_relaxed_node, replacement_node);
            
            return true;
        }
        
        // For other operations, we'll skip the conversion for now
        // This can be extended later as needed
        return false;
    };
    
    auto m = std::make_shared<ngraph::pattern::Matcher>(type_relaxed_op, matcher_name);
    this->register_matcher(m, callback);
}
