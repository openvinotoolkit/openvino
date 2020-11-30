//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <functional>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/util.hpp>
#include <ngraph/log.hpp>
#include <transformations/common_optimizations/nop_elimination.hpp>

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

#define TI(x) x::type_info

static bool eliminate_nop(const std::shared_ptr<Node>& node) {
    // skip if shapes are dynamic
    if (node->get_input_partial_shape(0).is_dynamic() ||
        node->get_output_partial_shape(0).is_dynamic()) {
        return false;
    }

    if (node->get_input_shape(0) == node->get_output_shape(0)) {
        return replace_output_update_name(node->output(0), node->input_value(0));
    }
    return false;
}

static bool eliminate_convert(const std::shared_ptr<Node>& node) {
    bool is_out_type_agnostic = false;
    static const std::set<NodeTypeInfo> type_agnostic{TI(opset3::NonZero)};
    if (node->output(0).get_target_inputs().size() == 1) {
        Input<Node> out = *node->output(0).get_target_inputs().begin();
        is_out_type_agnostic = type_agnostic.count(out.get_node()->get_type_info()) == 1;
    }
    auto convert = as_type_ptr<opset3::Convert>(node);
    auto input = convert->input_value(0);
    if (convert->get_convert_element_type() == input.get_element_type() || is_out_type_agnostic) {
        if (is_out_type_agnostic && is_type<opset3::Convert>(input.get_node())) {
            input = input.get_node()->input_value(0);
        }
        return replace_output_update_name(node->output(0), input);
    }
    return false;
}

static bool eliminate_concat(const std::shared_ptr<Node>& node) {
    auto node_input = node->input_value(0);

    // remove concat with single input
    if (node->get_input_size() == 1) {
        return replace_output_update_name(node->output(0), node_input);
    }
    return false;
}

static bool eliminate_reshape_v1(const std::shared_ptr<Node>& node) {
    auto input = node->input_value(0);
    // check if reshape is not identity op
    if (input.get_partial_shape().is_dynamic() || node->get_output_partial_shape(0).is_dynamic()) {
        NGRAPH_DEBUG << node << " has dynamic shapes.";
        return false;
    }
    // remove identity op
    if (input.get_shape() == node->get_output_shape(0)) {
        return replace_output_update_name(node->output(0), input);
    }
    // eliminate redundant reshape, squeeze, or unsqueeze
    auto input_node = input.get_node_shared_ptr();
    if (as_type_ptr<opset3::Squeeze>(input_node) ||
        as_type_ptr<opset3::Unsqueeze>(input_node) ||
        as_type_ptr<opset3::Reshape>(input_node)) {
        auto shape = node->get_output_shape(0);
        std::vector<int64_t> vi;
        vi.assign(shape.begin(), shape.end());
        auto pat = opset3::Constant::create<int64_t>(element::i64, Shape{vi.size()}, vi);
        auto new_reshape =
            make_shared<opset3::Reshape>(input.get_node()->input_value(0), pat, false);
        new_reshape->set_friendly_name(node->get_friendly_name());
        copy_runtime_info({input_node, node}, new_reshape);
        replace_node(node, new_reshape);
        return true;
    }

    return false;
}

static size_t count_unknown_dims(const PartialShape& ps) {
    size_t rc = 0;
    if (ps.is_static()) {
        return rc;
    }
    for (auto i = 0; i < ps.rank().get_length(); i++) {
        if (ps[i].is_dynamic()) {
            rc += 1;
        }
    }
    return rc;
}

static bool replace_squeeze_unsqueeze(const std::shared_ptr<Node>& node) {
    auto shape_ps = node->get_output_partial_shape(0);
    if (shape_ps.rank().get_length() == 0) {
        return false;
    }
    if (count_unknown_dims(shape_ps) > 1) {
        return false;
    }
    std::vector<int64_t> target_shape;
    for (auto i = 0; i < shape_ps.rank().get_length(); i++) {
        if (shape_ps[i].is_dynamic()) {
            target_shape.emplace_back(-1);
        } else {
            target_shape.emplace_back(shape_ps[i].get_length());
        }
    }

    shared_ptr<Node> reshape;
    auto input = node->input_value(0).get_node_shared_ptr();
    auto pat =
        opset3::Constant::create<int64_t>(element::i64, Shape{target_shape.size()}, target_shape);

    if (is_type<opset3::Reshape>(input) || is_type<opset3::Squeeze>(input) ||
        is_type<opset3::Unsqueeze>(input)) {
        reshape = make_shared<opset3::Reshape>(input->input_value(0), pat, false);
    } else {
        reshape = make_shared<opset3::Reshape>(node->input_value(0), pat, false);
    }

    // skip if reshape is nop
    if (reshape->get_input_partial_shape(0).same_scheme(shape_ps)) {
        return replace_output_update_name(node->output(0), reshape->input_value(0));
    } else {
        return replace_node_update_name(node, reshape);
    }
}

static std::vector<int64_t> get_unsqueeze_axes(const PartialShape& data_shape,
                                               const PartialShape& out_shape) {
    std::vector<int64_t> axes;
    int64_t i = 0;
    for (auto o = 0; o < out_shape.rank().get_length(); o++) {
        if (i < data_shape.rank().get_length() && data_shape[i].same_scheme(out_shape[o])) {
            i += 1;
            continue;
        }
        if (out_shape[o].is_static() && out_shape[o] == 1) {
            axes.push_back(o);
        }
    }
    return axes;
}

static std::vector<int64_t> get_squeeze_axes(const PartialShape& data_shape,
                                             const PartialShape& out_shape) {
    std::vector<int64_t> axes;
    int64_t out_i = 0;
    for (auto i = 0; i < data_shape.rank().get_length(); i++) {
        if (out_i < out_shape.rank().get_length() && data_shape[i].same_scheme(out_shape[out_i])) {
            out_i += 1;
            continue;
        }
        if (data_shape[i].is_static() && data_shape[i] == 1) {
            axes.push_back(i);
        }
    }
    return axes;
}

static bool eliminate_unsqueeze(const std::shared_ptr<Node>& node) {
    auto out_shape = node->get_output_partial_shape(0);
    // try to replace all squeeze/unsqueeze with reshape
    if (out_shape.rank().is_static() && out_shape.rank().get_length() != 0 && count_unknown_dims(out_shape) < 2) {
        return replace_squeeze_unsqueeze(node);
    }

    auto unsqueeze = as_type_ptr<opset3::Unsqueeze>(node);
    if (unsqueeze == nullptr)
        return false;
    auto input = unsqueeze->input_value(0).get_node_shared_ptr();
    auto squeeze = as_type_ptr<opset3::Squeeze>(input);
    auto replace_unsqueeze_only = [&](const vector<int64_t>& axes) {
        auto axes_const = opset3::Constant::create<int64_t>(element::i64, Shape{axes.size()}, axes);
        auto new_unsq = make_shared<opset3::Unsqueeze>(input->input_value(0), axes_const);
        if (unsqueeze->get_output_partial_shape(0).same_scheme(
                new_unsq->get_output_partial_shape(0))) {
            return replace_node_update_name(unsqueeze, new_unsq);
        }
        return false;
    };
    // eliminate redundant squeeze->unsqueeze
    if (squeeze) {
        const auto& data_shape = squeeze->input_value(0).get_partial_shape();
        if (ngraph::compare_constants(squeeze->input_value(1).get_node_shared_ptr(),
                                      unsqueeze->input_value(1).get_node_shared_ptr())) {
            return replace_output_update_name(unsqueeze->output(0), squeeze->input_value(0));
        }
        if (data_shape.rank().is_dynamic() || out_shape.rank().is_dynamic()) {
            return false;
        }
        if (out_shape.rank().get_length() > data_shape.rank().get_length()) {
            // check if single unsqueeze can handle this
            auto axes = get_unsqueeze_axes(data_shape, out_shape);
            if (axes.size() + data_shape.rank().get_length() == out_shape.rank().get_length()) {
                return replace_unsqueeze_only(axes);
            }
        }
        if (out_shape.rank().get_length() < data_shape.rank().get_length()) {
            // check if single squeeze can handle this
            auto axes = get_squeeze_axes(data_shape, out_shape);
            if (data_shape.rank().get_length() - axes.size() == out_shape.rank().get_length()) {
                auto axes_const =
                    opset3::Constant::create<int64_t>(element::i64, Shape{axes.size()}, axes);
                auto new_sq = make_shared<opset3::Squeeze>(input->input_value(0), axes_const);
                if (unsqueeze->get_output_partial_shape(0).same_scheme(
                        new_sq->get_output_partial_shape(0))) {
                    return replace_node_update_name(unsqueeze, new_sq);
                }
                return false;
            }
        }
        return false;
    }
    // eliminate redundant unsqueeze->unsqueeze
    auto unsqueeze_i = as_type_ptr<opset3::Unsqueeze>(input);
    if (unsqueeze_i) {
        const auto& data_shape = unsqueeze_i->input_value(0).get_partial_shape();
        if (data_shape.rank().is_dynamic() || out_shape.rank().is_dynamic()) {
            return false;
        }
        auto axes = get_unsqueeze_axes(data_shape, out_shape);
        return replace_unsqueeze_only(axes);
    }

    return false;
}

static bool eliminate_squeeze(const std::shared_ptr<Node>& node) {
    auto out_shape = node->get_output_partial_shape(0);
    // try to replace all unsqueeze/squeeze with reshape
    if (out_shape.rank().is_static() && out_shape.rank().get_length() != 0 && count_unknown_dims(out_shape) < 2) {
        return replace_squeeze_unsqueeze(node);
    }

    auto squeeze = as_type_ptr<opset3::Squeeze>(node);
    if (squeeze == nullptr)
        return false;
    auto input = squeeze->input_value(0).get_node_shared_ptr();
    auto replace_squeeze_only = [&](const vector<int64_t>& axes) {
        auto axes_const = opset3::Constant::create<int64_t>(element::i64, Shape{axes.size()}, axes);
        auto new_sq = make_shared<opset3::Squeeze>(input->input_value(0), axes_const);
        if (squeeze->get_output_partial_shape(0).same_scheme(new_sq->get_output_partial_shape(0))) {
            return replace_node_update_name(squeeze, new_sq);
        }
        return false;
    };
    // eliminate redundant unsqueeze->squeeze
    if (auto unsqueeze = as_type_ptr<opset3::Unsqueeze>(input)) {
        PartialShape data_shape;
        if (op::is_parameter(input)) {
            data_shape = unsqueeze->input(0).get_partial_shape();
        } else {
            data_shape = input->input(0).get_partial_shape();
        }
        if (ngraph::compare_constants(unsqueeze->input_value(1).get_node_shared_ptr(),
                                      squeeze->input_value(1).get_node_shared_ptr())) {
            return replace_output_update_name(squeeze->output(0), unsqueeze->input_value(0));
        }
        if (data_shape.rank().is_dynamic() || out_shape.rank().is_dynamic()) {
            return false;
        }
        if (out_shape.rank().get_length() < data_shape.rank().get_length()) {
            // check if single squeeze can handle this
            auto axes = get_squeeze_axes(data_shape, out_shape);
            if (data_shape.rank().get_length() == out_shape.rank().get_length() + axes.size()) {
                return replace_squeeze_only(axes);
            }
        }
        if (out_shape.rank().get_length() > data_shape.rank().get_length()) {
            // check if single unsqueeze can handle this
            auto axes = get_unsqueeze_axes(data_shape, out_shape);
            if (data_shape.rank().get_length() + axes.size() == out_shape.rank().get_length()) {
                auto axes_const =
                    opset3::Constant::create<int64_t>(element::i64, Shape{axes.size()}, axes);
                auto new_unsq = make_shared<opset3::Unsqueeze>(input->input_value(0), axes_const);
                if (squeeze->get_output_partial_shape(0).same_scheme(
                        new_unsq->get_output_partial_shape(0))) {
                    replace_output_update_name(squeeze, new_unsq);
                    return true;
                }
            }
        }
        return false;
    }
    // eliminate redundant squeeze->squeeze
    if (auto squeeze_i = as_type_ptr<opset3::Squeeze>(input)) {
        PartialShape data_shape;
        if (op::is_parameter(input)) {
            data_shape = squeeze_i->input(0).get_partial_shape();
        } else {
            data_shape = input->input(0).get_partial_shape();
        }
        if (data_shape.rank().is_dynamic() || out_shape.rank().is_dynamic()) {
            return false;
        }
        auto axes = get_squeeze_axes(data_shape, out_shape);
        return replace_squeeze_only(axes);
    }
    return false;
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::NopElimination, "NopElimination", 0);

bool pass::NopElimination::run_on_function(std::shared_ptr<Function> function) {
    static const std::unordered_map<NodeTypeInfo, std::function<bool(const std::shared_ptr<Node>&)>>
        dispatcher{{TI(opset3::Pad), &eliminate_nop},
                   {TI(opset3::Convert), &eliminate_convert},
                   {TI(opset3::Reshape), &eliminate_reshape_v1},
                   {TI(opset3::Concat), &eliminate_concat},
                   {TI(opset3::Squeeze), &eliminate_squeeze},
                   {TI(op::v1::Broadcast), &eliminate_nop},
                   {TI(opset3::Unsqueeze), &eliminate_unsqueeze}};

    bool clobbered = false;

    for (const auto& node : function->get_ops()) {
        // Recursively apply transformation for sub-graph based operations
        if (auto sub_graph_node = std::dynamic_pointer_cast<op::util::SubGraphOp>(node)) {
            if (auto sub_graph = sub_graph_node->get_function()) {
                clobbered |= run_on_function(sub_graph);
            }
        }
        auto handler = dispatcher.find(node->get_type_info());
        if (handler != dispatcher.end()) {
            clobbered |= handler->second(node);
        }
    }

    return clobbered;
}
