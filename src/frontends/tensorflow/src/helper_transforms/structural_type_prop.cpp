// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "structural_type_prop.hpp"

#include <memory>
#include <vector>
#include <numeric>

#include "openvino/frontend/tensorflow/frontend.hpp"
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "openvino/core/type/non_tensor_type.hpp"


namespace {
    // TODO: Remove this duplicate: CPU transforms has it, copied and pasted here

    bool is_data_movement_operation(const std::shared_ptr<ngraph::Node>& node) {
        return ov::is_type<ngraph::op::v0::Squeeze>(node) ||
               ov::is_type<ov::op::v1::StridedSlice>(node) ||
               ov::is_type<ngraph::op::v0::Unsqueeze>(node) ||
               ov::is_type<ngraph::op::v1::Reshape>(node) ||
               ov::is_type<ngraph::op::v1::Transpose>(node) ||
               ov::is_type<ngraph::op::v0::ShuffleChannels>(node) ||
               ov::is_type<ngraph::op::v7::Roll>(node) ||
               ov::is_type<ngraph::op::v0::ReverseSequence>(node) ||
               ov::is_type<ngraph::op::v0::DepthToSpace>(node) ||
               ov::is_type<ngraph::op::v1::BatchToSpace>(node) ||
               ov::is_type<ngraph::op::v1::Broadcast>(node) ||
               ov::is_type<ngraph::op::v3::Broadcast>(node) ||
               ov::is_type<ngraph::op::v1::Gather>(node) ||
               ov::is_type<ngraph::op::v7::Gather>(node) ||
               ov::is_type<ngraph::op::v8::Gather>(node) ||
               ov::is_type<ngraph::op::v0::Parameter>(node);
    }

    bool is_scalar_like(const std::shared_ptr<ngraph::Node>& node) {
        auto constantNode = std::dynamic_pointer_cast<ngraph::opset8::Constant>(node);
        return constantNode != nullptr && shape_size(constantNode->get_shape()) == 1;
    }
} // namespace

namespace ov {
namespace frontend {
namespace tensorflow {
namespace pass {

StructuralTypeProp::StructuralTypeProp() {
    auto data_movement = ngraph::pattern::wrap_type<ov::op::Op>(ov::pass::pattern::op::as_value_predicate(is_data_movement_operation));
    std::cerr << "[ INFO TF FE ] Registering StructuralTypeProp\n";

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto node = m.get_match_root();

        std::cerr << "[ INFO TF FE ] Matching data movement op: " << node->get_type_name() << "\n";

        // Depending on operation, propagate structural type field
        // TODO: This code should be moved to the operations themselves, but now we are trying
        // to avoid any impact on OV structures and implement it externally.
        // Code amount required to implement it in core will be similar to what we are doing
        // here except we won't have similar mega-switches based on op types.

        if (auto parameter = std::dynamic_pointer_cast<ov::op::v0::Parameter>(node)) {
            // Parameter should have a special RT info attribute `structural_type` that should be copied
            // to the output tensor rt_info

            std::cerr << "[ INFO TF FE ] Detected Parameter\n";

            StructuralTypeAttribute::copy(parameter->get_rt_info(), parameter->get_output_tensor(0).get_rt_info());
        } else if (auto reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(node)) {
            std::cerr << "[ INFO TF FE ] Detected Reshape\n";
            StructuralTypeAttribute::copy(reshape->get_input_tensor(0).get_rt_info(), reshape->get_output_tensor(0).get_rt_info());
        }

        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(data_movement, "ov::frontend::tensorflow::pass::StructuralTypeProp");
    register_matcher(m, callback);
}


ReplaceStrByU81D::ReplaceStrByU81D() {
    auto str_tensor = ngraph::pattern::wrap_type<ov::op::Op>(
        ov::pass::pattern::op::ValuePredicate([](ov::Output<ov::Node> x) {
            std::cerr << "get_rt_info: " << x.get_tensor().get_rt_info().size() << "\n";
            //return false;
            std::cerr.flush();
            return StructuralTypeAttribute::has_type(x.get_tensor().get_rt_info(), element::StructuralType::Str());
            // FIXME: Check that this is a scalar, otherwise this transformation doesn't work
            // FIXME: For now we re-interpret all tensors that have Str type as a scalar tensors
        }));

    std::cerr << "[ INFO TF FE ] Registering ReplaceStrByU81D\n";

    auto callback = [](ov::pass::pattern::Matcher& m) {
        //return false;
        auto port = m.get_match_value();  // TODO: iterate over all outputs and check each of it to match the criteria
        auto node = m.get_match_root();

        std::cerr << "[ INFO TF FE ] Detected tensor with Str type: " << node->get_type_name() << "\n";

        if (auto parameter = std::dynamic_pointer_cast<ov::op::v0::Parameter>(node)) {
            std::cerr << "Parameter override to u8/1d\n";
            parameter->set_element_type(element::u8);
            parameter->set_partial_shape(PartialShape{Dimension()});
        }

        // Just setting type and shape without shape propagation -- will require re-validation of the function
        // in the end to catch all inconsistencies due to possible bugs.

        port.get_tensor().set_tensor_type(element::u8, PartialShape{Dimension()});
        //std::cerr << "move to original\n";
        //StructuralTypeAttribute::move_to_original(port.get_tensor().get_rt_info());
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(str_tensor, "ov::frontend::tensorflow::pass::ReplaceStrByU81D");
    register_matcher(m, callback);
}



}
}
}
}
