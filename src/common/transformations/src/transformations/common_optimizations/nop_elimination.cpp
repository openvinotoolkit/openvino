// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>
#include <memory>
#include <ngraph/log.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/util.hpp>
#include <numeric>
#include <transformations/common_optimizations/nop_elimination.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"

using namespace std;
using namespace ngraph;

//`simplify_gather`, optimizes gather if Gather is gathering the
// whole input tensor
static bool simplify_gather(std::shared_ptr<Node> node) {
    if (auto gather = ov::as_type_ptr<opset3::Gather>(node)) {
        // check if we are gathering the whole input
        auto data = gather->input_value(0);
        auto indices = gather->input_value(1);

        // we need to know data and indices shape to infer if gather is Nop
        if (data.get_partial_shape().is_dynamic() || indices.get_partial_shape().is_dynamic()) {
            return false;
        }
        // if rank of data and gather output dont match, we will skip
        if (data.get_shape().size() != node->get_shape().size()) {
            return false;
        }

        auto axis = gather->get_axis();
        if (axis == opset3::Gather::AXIS_NOT_SET_VALUE) {
            NGRAPH_DEBUG << "axis value not set";
            return false;
        }

        // case_1 : if the input tensor is of shape (4, 1, 4)
        // and axis = 1, then the gather would be simply
        // gathering the whole input tensor, so we can optimize this
        // op has Nop

        if (data.get_shape()[axis] == 1 && data.get_shape() == node->get_shape()) {
            return replace_output_update_name(gather->output(0), gather->input_value(0));
        }

        // case_2 : if the input tensor is of shape (4, 3, 4)
        // we need to check the contents of indices, if indices
        // is 1D tensor of value {0, 1, 2}, we can optimize this
        // op has Nop

        // check if the indices is constant
        auto constant_indices = ov::as_type_ptr<opset3::Constant>(gather->input_value(1).get_node_shared_ptr());
        if (!constant_indices) {
            return false;
        } else {
            // if ref_inidices == indices, we are capturing the
            // entire input tensor
            std::vector<int64_t> ref_indices(data.get_shape()[axis], 0);
            std::iota(ref_indices.begin(), ref_indices.end(), 0);
            if (ref_indices == constant_indices->cast_vector<int64_t>()) {
                return replace_output_update_name(gather->output(0), gather->input_value(0));
            }
        }
    }
    return false;
}

static bool eliminate_nop(const std::shared_ptr<Node>& node) {
    // skip if shapes are dynamic
    if (node->get_input_partial_shape(0).is_dynamic() || node->get_output_partial_shape(0).is_dynamic()) {
        return false;
    }

    if (node->get_input_shape(0) == node->get_output_shape(0)) {
        return replace_output_update_name(node->output(0), node->input_value(0));
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
    if (ov::as_type_ptr<opset3::Squeeze>(input_node) || ov::as_type_ptr<opset3::Unsqueeze>(input_node) ||
        ov::as_type_ptr<opset3::Reshape>(input_node)) {
        if (input_node->get_output_target_inputs(0).size() != 1)
            return false;

        auto shape = node->get_output_shape(0);

        // remove interchangeable nodes
        if (input_node->get_input_partial_shape(0).is_static() && input_node->get_input_shape(0) == shape) {
            return replace_output_update_name(node->output(0), input_node->input_value(0));
        } else {
            std::vector<int64_t> vi;
            vi.assign(shape.begin(), shape.end());
            auto pat = opset3::Constant::create<int64_t>(element::i64, Shape{vi.size()}, vi);
            auto new_reshape = make_shared<opset3::Reshape>(input.get_node()->input_value(0), pat, false);
            new_reshape->set_friendly_name(node->get_friendly_name());
            copy_runtime_info({input_node, node}, new_reshape);
            replace_node(node, new_reshape);
            return true;
        }
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
    auto pat = opset3::Constant::create<int64_t>(element::i64, Shape{target_shape.size()}, target_shape);

    if (ov::is_type<opset3::Reshape>(input) || ov::is_type<opset3::Squeeze>(input) ||
        ov::is_type<opset3::Unsqueeze>(input)) {
        reshape = make_shared<opset3::Reshape>(input->input_value(0), pat, false);
    } else {
        reshape = make_shared<opset3::Reshape>(node->input_value(0), pat, false);
    }

    // skip if reshape is nop
    if (reshape->get_input_partial_shape(0).same_scheme(shape_ps)) {
        copy_runtime_info({input, node->output(0).get_node_shared_ptr()}, node->output(0).get_node_shared_ptr());
        return replace_output_update_name(node->output(0), reshape->input_value(0));
    } else {
        return replace_node_update_name(node, reshape);
    }
}

static std::vector<int64_t> get_unsqueeze_axes(const PartialShape& data_shape, const PartialShape& out_shape) {
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

static std::vector<int64_t> get_squeeze_axes(const PartialShape& data_shape, const PartialShape& out_shape) {
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

    auto unsqueeze = ov::as_type_ptr<opset3::Unsqueeze>(node);
    if (unsqueeze == nullptr)
        return false;
    auto input = unsqueeze->input_value(0).get_node_shared_ptr();
    auto squeeze = ov::as_type_ptr<opset3::Squeeze>(input);
    auto replace_unsqueeze_only = [&](const vector<int64_t>& axes) {
        auto axes_const = opset3::Constant::create<int64_t>(element::i64, Shape{axes.size()}, axes);
        auto new_unsq = make_shared<opset3::Unsqueeze>(input->input_value(0), axes_const);
        if (unsqueeze->get_output_partial_shape(0).same_scheme(new_unsq->get_output_partial_shape(0))) {
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
            if (static_cast<int64_t>(axes.size()) + data_shape.rank().get_length() == out_shape.rank().get_length()) {
                return replace_unsqueeze_only(axes);
            }
        }
        if (out_shape.rank().get_length() < data_shape.rank().get_length()) {
            // check if single squeeze can handle this
            auto axes = get_squeeze_axes(data_shape, out_shape);
            if (data_shape.rank().get_length() - static_cast<int64_t>(axes.size()) == out_shape.rank().get_length()) {
                auto axes_const = opset3::Constant::create<int64_t>(element::i64, Shape{axes.size()}, axes);
                auto new_sq = make_shared<opset3::Squeeze>(input->input_value(0), axes_const);
                if (unsqueeze->get_output_partial_shape(0).same_scheme(new_sq->get_output_partial_shape(0))) {
                    return replace_node_update_name(unsqueeze, new_sq);
                }
                return false;
            }
        }
        return false;
    }
    // eliminate redundant unsqueeze->unsqueeze
    auto unsqueeze_i = ov::as_type_ptr<opset3::Unsqueeze>(input);
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

#define ECHO(NAME) #NAME
#define STR(NAME)  ECHO(NAME)
#define SIMPLE_MATCHER_PASS_DEFINITION(NAME, OP, FUNC)                                     \
    class NAME : public ngraph::pass::MatcherPass {                                        \
    public:                                                                                \
        OPENVINO_RTTI(STR(NAME), "0");                                                     \
        NAME() {                                                                           \
            MATCHER_SCOPE(NAME);                                                           \
            auto match_node = ngraph::pattern::wrap_type<OP>();                            \
            ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {    \
                MATCHER_SCOPE_ENABLE(NAME);                                                \
                return FUNC(m.get_match_root());                                           \
            };                                                                             \
            auto m = std::make_shared<ngraph::pattern::Matcher>(match_node, matcher_name); \
            register_matcher(m, callback);                                                 \
        }                                                                                  \
    };

SIMPLE_MATCHER_PASS_DEFINITION(EliminateReshape, opset3::Reshape, eliminate_reshape_v1);
SIMPLE_MATCHER_PASS_DEFINITION(EliminateUnsqueeze, opset3::Unsqueeze, eliminate_unsqueeze);
SIMPLE_MATCHER_PASS_DEFINITION(EliminateBroadcast, op::v1::Broadcast, eliminate_nop);
SIMPLE_MATCHER_PASS_DEFINITION(EliminateGather, opset3::Gather, simplify_gather);

pass::EliminatePad::EliminatePad() {
    MATCHER_SCOPE(EliminatePad);
    auto pad_node_pattern = pattern::wrap_type<opset8::Pad>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto pad = m.get_match_root();

        auto pad_begin_const = ngraph::get_constant_from_source(pad->input_value(1));
        auto pad_end_const = ngraph::get_constant_from_source(pad->input_value(2));

        if (!pad_begin_const || !pad_end_const) {
            return false;
        }

        const auto pad_begin_value = pad_begin_const->cast_vector<int64_t>();
        const auto pad_end_value = pad_end_const->cast_vector<int64_t>();

        if (std::any_of(pad_begin_value.begin(),
                        pad_begin_value.end(),
                        [](int64_t value) {
                            return value != 0;
                        }) ||
            std::any_of(pad_end_value.begin(), pad_end_value.end(), [](int64_t value) {
                return value != 0;
            })) {
            return false;
        }
        MATCHER_SCOPE_ENABLE(EliminatePad);
        return replace_output_update_name(pad->output(0), pad->input_value(0));
    };

    auto m = std::make_shared<pattern::Matcher>(pad_node_pattern, matcher_name);
    this->register_matcher(m, callback);
}

pass::EliminateConvert::EliminateConvert() {
    MATCHER_SCOPE(EliminateConvert);
    auto convert_pattern = pattern::wrap_type<opset8::Convert>();

    matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto convert = std::dynamic_pointer_cast<opset8::Convert>(m.get_match_root());
        if (!convert) {
            return false;
        }
        if (convert->get_input_element_type(0) == convert->get_element_type()) {
            MATCHER_SCOPE_ENABLE(EliminateConvert);
            return replace_output_update_name(convert->output(0), convert->input_value(0));
        }
        return false;
    };

    auto m = std::make_shared<pattern::Matcher>(convert_pattern, matcher_name);
    this->register_matcher(m, callback);
}

pass::EliminateConvertNonZero::EliminateConvertNonZero() {
    MATCHER_SCOPE(EliminateConvertNonZero);
    auto convert_pattern = pattern::wrap_type<opset8::Convert>(pattern::consumers_count(1));
    auto non_zero = pattern::wrap_type<opset8::NonZero>({convert_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_map();
        auto convert = pattern_map.at(convert_pattern);
        // remove convert
        convert->output(0).replace(convert->input_value(0));
        // to make this elimination recursive we register NonZero as a node which will be used to repeat matching
        register_new_node(m.get_match_root());
        MATCHER_SCOPE_ENABLE(EliminateConvertNonZero);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(non_zero, matcher_name);
    this->register_matcher(m, callback);
}

pass::EliminateConcat::EliminateConcat() {
    MATCHER_SCOPE(EliminateConcat);
    auto convert_pattern = pattern::wrap_type<opset8::Concat>();

    matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto concat = m.get_match_root();
        if (concat->inputs().size() == 1) {
            MATCHER_SCOPE_ENABLE(EliminateConcat);
            return replace_output_update_name(concat->output(0), concat->input_value(0));
        }
        return false;
    };

    auto m = std::make_shared<pattern::Matcher>(convert_pattern, matcher_name);
    this->register_matcher(m, callback);
}

pass::EliminateSplit::EliminateSplit() {
    MATCHER_SCOPE(EliminateConcat);
    auto convert_pattern = pattern::wrap_type<opset8::Split>();

    matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto split = std::dynamic_pointer_cast<opset8::Split>(m.get_match_root());
        if (!split || split->get_num_splits() != 1) {
            return false;
        }
        MATCHER_SCOPE_ENABLE(EliminateConcat);
        return replace_output_update_name(split->output(0), split->input_value(0));
    };

    auto m = std::make_shared<pattern::Matcher>(convert_pattern, matcher_name);
    this->register_matcher(m, callback);
}

pass::EliminateSqueeze::EliminateSqueeze() {
    MATCHER_SCOPE(EliminateSqueeze);
    auto squeeze_pattern = pattern::wrap_type<opset8::Squeeze>();

    matcher_pass_callback callback = [](pattern::Matcher& m) {
        MATCHER_SCOPE_ENABLE(EliminateSqueeze);
        const auto node = m.get_match_root();
        auto out_shape = node->get_output_partial_shape(0);
        // try to replace all unsqueeze/squeeze with reshape
        if (out_shape.rank().is_static() && out_shape.rank().get_length() != 0 && count_unknown_dims(out_shape) < 2) {
            return replace_squeeze_unsqueeze(node);
        }

        auto squeeze = ov::as_type_ptr<opset3::Squeeze>(node);
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
        if (auto unsqueeze = ov::as_type_ptr<opset3::Unsqueeze>(input)) {
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
                if (data_shape.rank().get_length() ==
                    out_shape.rank().get_length() + static_cast<int64_t>(axes.size())) {
                    return replace_squeeze_only(axes);
                }
            }
            if (out_shape.rank().get_length() > data_shape.rank().get_length()) {
                // check if single unsqueeze can handle this
                auto axes = get_unsqueeze_axes(data_shape, out_shape);
                if (data_shape.rank().get_length() + static_cast<int64_t>(axes.size()) ==
                    out_shape.rank().get_length()) {
                    auto axes_const = opset3::Constant::create<int64_t>(element::i64, Shape{axes.size()}, axes);
                    auto new_unsq = make_shared<opset3::Unsqueeze>(input->input_value(0), axes_const);
                    if (squeeze->get_output_partial_shape(0).same_scheme(new_unsq->get_output_partial_shape(0))) {
                        replace_output_update_name(squeeze, new_unsq);
                        return true;
                    }
                }
            }
            return false;
        }
        // eliminate redundant squeeze->squeeze
        if (auto squeeze_i = ov::as_type_ptr<opset3::Squeeze>(input)) {
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
    };

    auto m = std::make_shared<pattern::Matcher>(squeeze_pattern, matcher_name);
    this->register_matcher(m, callback);
}

pass::EliminateTranspose::EliminateTranspose() {
    MATCHER_SCOPE(EliminateTranspose);
    auto order = pattern::wrap_type<opset8::Constant>();
    auto transpose_pattern = pattern::wrap_type<opset8::Transpose>({pattern::any_input(), order});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_map();
        auto order_const = std::dynamic_pointer_cast<opset8::Constant>(pattern_map.at(order));
        if (!order_const) {
            return false;
        }

        const auto& order_values = order_const->cast_vector<int64_t>();
        vector<int64_t> ref_values(order_values.size());
        std::iota(ref_values.begin(), ref_values.end(), 0);
        if (order_values != ref_values) {
            return false;
        }

        auto transpose = m.get_match_root();
        MATCHER_SCOPE_ENABLE(EliminateTranspose);
        return replace_output_update_name(transpose->output(0), transpose->input_value(0));
    };

    auto m = std::make_shared<pattern::Matcher>(transpose_pattern, matcher_name);
    this->register_matcher(m, callback);
}

pass::EliminateEltwise::EliminateEltwise() {
    MATCHER_SCOPE(EliminateEltwise);
    auto input = pattern::any_input();
    auto constant_pattern = pattern::wrap_type<opset8::Constant>();
    auto eltwise_pattern =
        pattern::wrap_type<opset8::Add, opset8::Subtract, opset8::Multiply, opset8::Divide>({input, constant_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto eltwise = pattern_map.at(eltwise_pattern).get_node_shared_ptr();
        auto non_const_input = pattern_map.at(input);
        auto constant = pattern_map.at(constant_pattern);

        if (!op::util::can_eliminate_eltwise_node(eltwise, constant, non_const_input)) {
            return false;
        }
        MATCHER_SCOPE_ENABLE(EliminateEltwise);
        return replace_output_update_name(eltwise->output(0), non_const_input);
    };

    auto m = std::make_shared<pattern::Matcher>(eltwise_pattern, matcher_name);
    this->register_matcher(m, callback);
}

ngraph::pass::NopElimination::NopElimination(bool use_shape_for_elimination) {
    // shape-agnostic transformations
    add_matcher<EliminatePad>();
    add_matcher<EliminateConvert>();
    add_matcher<EliminateConvertNonZero>();
    add_matcher<EliminateConcat>();
    add_matcher<EliminateSplit>();
    add_matcher<EliminateTranspose>();
    add_matcher<EliminateEltwise>();

    // shape-dependent transformations
    if (use_shape_for_elimination) {
        add_matcher<EliminateReshape>();
        add_matcher<EliminateSqueeze>();
        add_matcher<EliminateUnsqueeze>();
        add_matcher<EliminateBroadcast>();
        add_matcher<EliminateGather>();
    }
}
