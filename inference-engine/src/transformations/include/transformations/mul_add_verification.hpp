// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ie_api.h>

#include <ngraph/ngraph.hpp>

#include "ngraph/pattern/matcher.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/experimental/dyn_broadcast.hpp"
#include "ngraph/op/fused/conv_fused.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/add.hpp"

#include "ngraph/op/fused/group_conv.hpp"

#include <ngraph/pass/graph_rewrite.hpp>

#include "mul_add_squence_fusion.hpp"
#include <transformations/utils/annotations.hpp>

namespace ngraph {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(MulAddVerification);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::MulAddVerification: public ngraph::pass::GraphRewrite {
public:
    /*
     * This transformation aligns all Multiply and Add operations to have the same order of inputs
     * In case if one of inputs is Constant it should be placed to the second input
     */
    MulAddVerification() : GraphRewrite() {
        mul_add_verification<ngraph::op::v1::Add>();
        mul_add_verification<ngraph::op::v1::Multiply>();
    }

private:
    template<class T>
    void mul_add_verification();
};

template<class T>
void ngraph::pass::MulAddVerification::mul_add_verification() {
    Shape shape{};
    auto input1 = make_shared<pattern::op::Label>(element::f32, shape);
    auto input2 = make_shared<pattern::op::Label>(element::f32, shape);
    auto eltwise = make_shared<T>(input1, input2);

    ngraph::graph_rewrite_callback callback = [](ngraph::pattern::Matcher &m) {
        if (auto eltwise = dynamic_pointer_cast<T>(m.get_match_root())) {
            auto in0 = std::dynamic_pointer_cast<op::Constant>(eltwise->input(0).get_source_output().get_node_shared_ptr());
            auto in1 = std::dynamic_pointer_cast<op::Constant>(eltwise->input(1).get_source_output().get_node_shared_ptr());

            auto attrs = make_shared<ngraph::op::util::EltwiseAttrs>();
            if (in0) {
                attrs->set_const_input_id(0);
            }

            if (in1) {
                attrs->set_const_input_id(1);
            }

            attrs->set_consumers_count(eltwise->output(0).get_target_inputs().size());

            eltwise->set_op_annotations(attrs);
            return true;
        }

        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(eltwise, "MulAddVerification");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
