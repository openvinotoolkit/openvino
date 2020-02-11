// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <typeindex>

#include <ngraph/pass/graph_rewrite.hpp>

#include <ngraph/type/bfloat16.hpp>
#include <ngraph/type/float16.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/reshape.hpp>

#include "ngraph/runtime/reference/reshape.hpp"

namespace ngraph {
namespace pass {

class ReshapeConstanFolding;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ReshapeConstanFolding: public ngraph::pass::GraphRewrite {
public:
    ReshapeConstanFolding() : GraphRewrite() {
        reshape_constant_folding();
    }

private:
    template <class T>
    std::shared_ptr<ngraph::op::Constant> fold_constant_reshape(std::shared_ptr<ngraph::op::Constant> constant,
            std::shared_ptr<ngraph::op::Reshape> reshape,
            ngraph::NodeExecutorTy func) {
        auto out_shape = reshape->get_shape();
        std::vector<T> out_vec(shape_size(out_shape));

        if (func != nullptr) {
            std::vector<void*> inputs;
            inputs.push_back(const_cast<void*>(constant->get_data_ptr()));
            std::vector<void*> outputs;
            outputs.push_back(out_vec.data());

            func(inputs, outputs);
        } else {
            ngraph::runtime::reference::reshape<T>(constant->get_data_ptr<T>(),
                    out_vec.data(),
                    constant->get_shape(),
                    reshape->get_input_order(),
                    out_shape);
        }

        return std::make_shared<ngraph::op::Constant>(constant->get_element_type(), out_shape, out_vec);
    }

    void reshape_constant_folding() {
        auto constant_label = std::make_shared<pattern::op::Label>(
                element::f32, Shape{2, 4}, pattern::has_class<op::Constant>());
        auto reshape = std::make_shared<op::Reshape>(constant_label, AxisVector{0, 1}, Shape{2, 4, 1});

        auto constant_reshape_callback = [&, constant_label](pattern::Matcher& m) {
            const ngraph::BuildNodeExecutorMap m_cfmap = ngraph::BuildNodeExecutorMap();
            auto pattern_map = m.get_pattern_map();

            auto constant_match = std::static_pointer_cast<op::Constant>(pattern_map[constant_label]);
            auto reshape_match = std::static_pointer_cast<op::Reshape>(m.get_match_root());

            NodeExecutorTy func = nullptr;
            if (!m_cfmap.empty()) {
                auto handler = m_cfmap.find(std::type_index(typeid(ngraph::op::Reshape)));
                func = handler->second(reshape_match.get());
            }

            auto type = constant_match->get_element_type();
            if (type == element::i32) {
                replace_node(m.get_match_root(),
                        fold_constant_reshape<int>(constant_match, reshape_match, func));
                return true;
            } else if (type == element::i8) {
                replace_node(m.get_match_root(),
                        fold_constant_reshape<int8_t>(constant_match, reshape_match, func));
                return true;
            } else if (type == element::f32) {
                replace_node(m.get_match_root(),
                        fold_constant_reshape<float>(constant_match, reshape_match, func));
                return true;
            } else if (type == element::f64) {
                replace_node(m.get_match_root(),
                        fold_constant_reshape<double>(constant_match, reshape_match, func));
                return true;
            } else if (type == element::f16) {
                replace_node(
                        m.get_match_root(),
                        fold_constant_reshape<ngraph::float16>(constant_match, reshape_match, func));
                return true;
            } else if (type == element::bf16) {
                replace_node(
                        m.get_match_root(),
                        fold_constant_reshape<ngraph::bfloat16>(constant_match, reshape_match, func));
                return true;
            }

            return false;
        };

        auto reshape_matcher = std::make_shared<pattern::Matcher>(reshape, "ConstantFolding.ConstantReshape");
        this->add_matcher(reshape_matcher, constant_reshape_callback, PassProperty::CHANGE_DYNAMIC_STATE);
    }
};
