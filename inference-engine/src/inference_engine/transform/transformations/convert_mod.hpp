// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ngraph/pass/graph_rewrite.hpp>

#include <ngraph/op/fused/mod.hpp>

namespace ngraph {
namespace pass {

class ConvertMod;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertMod: public ngraph::pass::GraphRewrite {
public:
    ConvertMod() : GraphRewrite() {
        convert_mod();
    }

private:
    void convert_mod() {
        auto input0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
        auto input1 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
        auto mod = std::make_shared<ngraph::op::v1::Mod>(input0, input1);

        ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
            auto mod = std::dynamic_pointer_cast<ngraph::op::v1::Mod> (m.get_match_root());
            if (!mod) {
                return false;
            }
            auto last_node = mod->decompose_op()[0];
            last_node->set_friendly_name(mod->get_friendly_name());
            ngraph::replace_node(mod, last_node);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(mod, "ConvertMod");
        this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
    }
};
