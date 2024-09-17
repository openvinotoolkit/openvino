// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/nop_elimination.hpp"

#include <functional>
#include <memory>
#include <numeric>

#include "compare.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/gru_sequence.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/rnn_sequence.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/arithmetic_reductions_keep_dims.hpp"
#include "openvino/op/util/binary_elementwise_comparison.hpp"
#include "openvino/op/util/binary_elementwise_logical.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/op/util/pad_base.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/log.hpp"
#include "openvino/util/util.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;
using namespace ov;

//`simplify_gather`, optimizes gather if Gather is gathering the
// whole input tensor
static bool simplify_gather(shared_ptr<Node> node) {
    if (auto gather = ov::as_type_ptr<op::util::GatherBase>(node)) {
        // check if we are gathering the whole input
        auto data = gather->input_value(0);
        auto indices = gather->input_value(1);

        // we need to know data and indices shape to infer if gather is Nop
        if (data.get_partial_shape().is_dynamic() || indices.get_partial_shape().is_dynamic()) {
            return false;
        }

        auto axis = gather->get_axis();
        if (axis == ov::op::v1::Gather::AXIS_NOT_SET_VALUE) {
            OPENVINO_DEBUG("axis value not set");
            return false;
        }

        if (data.get_shape().size() != node->get_shape().size()) {
            auto constant_indices = ov::as_type_ptr<ov::op::v0::Constant>(gather->input_value(1).get_node_shared_ptr());
            if (!constant_indices)
                return false;
            // case_3: if input_shape is (1,3,5,5) and axis = 0, indices = 0, then gather is just a Squeeze
            const auto constant_indices_size = constant_indices->get_output_shape(0).size();
            const auto const_indices = constant_indices->cast_vector<int64_t>();
            if (data.get_shape()[axis] == 1 && (constant_indices_size == 0 || constant_indices_size == 1) &&
                const_indices[0] == 0) {
                auto squeeze = std::make_shared<ov::op::v0::Squeeze>(gather->input_value(0), gather->input_value(2));
                squeeze->set_friendly_name(gather->get_friendly_name());
                ov::copy_runtime_info(gather, squeeze);
                ov::replace_node(gather, squeeze);
                return true;
            }
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
        auto constant_indices = ov::as_type_ptr<ov::op::v0::Constant>(gather->input_value(1).get_node_shared_ptr());
        if (!constant_indices) {
            return false;
        } else {
            // if ref_inidices == indices, we are capturing the
            // entire input tensor
            vector<int64_t> ref_indices(data.get_shape()[axis], 0);
            iota(ref_indices.begin(), ref_indices.end(), 0);
            if (ref_indices == constant_indices->cast_vector<int64_t>()) {
                return replace_output_update_name(gather->output(0), gather->input_value(0));
            }
        }
    }
    return false;
}

static bool eliminate_nop(const shared_ptr<Node>& node) {
    // skip if shapes are dynamic
    if (node->get_input_partial_shape(0).is_dynamic() || node->get_output_partial_shape(0).is_dynamic()) {
        return false;
    }

    if (node->get_input_shape(0) == node->get_output_shape(0)) {
        return replace_output_update_name(node->output(0), node->input_value(0));
    }
    return false;
}

static bool eliminate_reshape_v1(const shared_ptr<Node>& node) {
    auto input = node->input_value(0);

    if (input.get_partial_shape().rank().is_static() && input.get_partial_shape().rank().same_scheme(1)) {
        if (input.get_partial_shape().same_scheme(node->get_output_partial_shape(0)))
            return replace_output_update_name(node->output(0), input);
    }

    // check if reshape is not identity op
    if (input.get_partial_shape().is_dynamic() || node->get_output_partial_shape(0).is_dynamic()) {
        OPENVINO_DEBUG(node, " has dynamic shapes.");
        return false;
    }
    // remove identity op
    if (input.get_shape() == node->get_output_shape(0)) {
        return replace_output_update_name(node->output(0), input);
    }
    // eliminate redundant reshape, squeeze, or unsqueeze
    auto input_node = input.get_node_shared_ptr();
    if (ov::as_type_ptr<ov::op::v0::Squeeze>(input_node) || ov::as_type_ptr<ov::op::v0::Unsqueeze>(input_node) ||
        ov::as_type_ptr<ov::op::v1::Reshape>(input_node)) {
        if (input_node->get_output_target_inputs(0).size() != 1)
            return false;

        auto shape = node->get_output_shape(0);

        // remove interchangeable nodes
        if (input_node->get_input_partial_shape(0).is_static() && input_node->get_input_shape(0) == shape) {
            return replace_output_update_name(node->output(0), input_node->input_value(0));
        } else {
            vector<int64_t> vi;
            vi.assign(shape.begin(), shape.end());
            auto pat = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{vi.size()}, vi);
            auto new_reshape = make_shared<ov::op::v1::Reshape>(input.get_node()->input_value(0), pat, false);
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

static bool replace_squeeze_unsqueeze(const shared_ptr<Node>& node) {
    auto shape_ps = node->get_output_partial_shape(0);
    if (shape_ps.rank().get_length() == 0) {
        return false;
    }
    if (count_unknown_dims(shape_ps) > 1) {
        return false;
    }
    vector<int64_t> target_shape;
    for (auto i = 0; i < shape_ps.rank().get_length(); i++) {
        if (shape_ps[i].is_dynamic()) {
            target_shape.emplace_back(-1);
        } else {
            target_shape.emplace_back(shape_ps[i].get_length());
        }
    }

    shared_ptr<Node> reshape;
    auto input = node->input_value(0).get_node_shared_ptr();
    auto pat = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{target_shape.size()}, target_shape);

    if (ov::is_type<ov::op::v1::Reshape>(input) || ov::is_type<ov::op::v0::Squeeze>(input) ||
        ov::is_type<ov::op::v0::Unsqueeze>(input)) {
        reshape = make_shared<ov::op::v1::Reshape>(input->input_value(0), pat, false);
    } else {
        reshape = make_shared<ov::op::v1::Reshape>(node->input_value(0), pat, false);
    }

    // skip if reshape is nop
    if (reshape->get_input_partial_shape(0).same_scheme(shape_ps)) {
        copy_runtime_info({input, node->output(0).get_node_shared_ptr()}, node->output(0).get_node_shared_ptr());
        return replace_output_update_name(node->output(0), reshape->input_value(0));
    } else {
        return replace_node_update_name(node, reshape);
    }
}

static vector<int64_t> get_unsqueeze_axes(const PartialShape& data_shape, const PartialShape& out_shape) {
    vector<int64_t> axes;
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

static vector<int64_t> get_squeeze_axes(const PartialShape& data_shape, const PartialShape& out_shape) {
    vector<int64_t> axes;
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

static bool eliminate_unsqueeze(const shared_ptr<Node>& node) {
    auto out_shape = node->get_output_partial_shape(0);
    // try to replace all squeeze/unsqueeze with reshape
    if (out_shape.rank().is_static() && out_shape.rank().get_length() != 0 && count_unknown_dims(out_shape) < 2) {
        return replace_squeeze_unsqueeze(node);
    }

    auto unsqueeze = ov::as_type_ptr<ov::op::v0::Unsqueeze>(node);
    if (unsqueeze == nullptr)
        return false;
    auto input = unsqueeze->input_value(0).get_node_shared_ptr();
    auto squeeze = ov::as_type_ptr<ov::op::v0::Squeeze>(input);
    auto replace_unsqueeze_only = [&](const vector<int64_t>& axes) {
        auto axes_const = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{axes.size()}, axes);
        auto new_unsq = make_shared<ov::op::v0::Unsqueeze>(input->input_value(0), axes_const);
        if (unsqueeze->get_output_partial_shape(0).same_scheme(new_unsq->get_output_partial_shape(0))) {
            return replace_node_update_name(unsqueeze, new_unsq);
        }
        return false;
    };
    // eliminate redundant squeeze->unsqueeze
    if (squeeze) {
        const auto& data_shape = squeeze->input_value(0).get_partial_shape();
        if (squeeze->inputs().size() > 1 && ov::compare_constants(squeeze->input_value(1).get_node_shared_ptr(),
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
                auto axes_const = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{axes.size()}, axes);
                auto new_sq = make_shared<ov::op::v0::Squeeze>(input->input_value(0), axes_const);
                if (unsqueeze->get_output_partial_shape(0).same_scheme(new_sq->get_output_partial_shape(0))) {
                    return replace_node_update_name(unsqueeze, new_sq);
                }
                return false;
            }
        }
        return false;
    }
    // eliminate redundant unsqueeze->unsqueeze
    auto unsqueeze_i = ov::as_type_ptr<ov::op::v0::Unsqueeze>(input);
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
#define SIMPLE_MATCHER_PASS_DEFINITION(NAME, FUNC, ...)                                 \
    class NAME : public ov::pass::MatcherPass {                                         \
    public:                                                                             \
        OPENVINO_RTTI(STR(NAME), "0");                                                  \
        NAME() {                                                                        \
            MATCHER_SCOPE(NAME);                                                        \
            auto match_node = ov::pass::pattern::wrap_type<__VA_ARGS__>();              \
            ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {   \
                return FUNC(m.get_match_root());                                        \
            };                                                                          \
            auto m = make_shared<ov::pass::pattern::Matcher>(match_node, matcher_name); \
            register_matcher(m, callback);                                              \
        }                                                                               \
    };

SIMPLE_MATCHER_PASS_DEFINITION(EliminateReshape, eliminate_reshape_v1, ov::op::v1::Reshape);
SIMPLE_MATCHER_PASS_DEFINITION(EliminateBroadcast, eliminate_nop, op::v1::Broadcast, op::v3::Broadcast);
SIMPLE_MATCHER_PASS_DEFINITION(EliminateGather,
                               simplify_gather,
                               ov::op::v1::Gather,
                               ov::op::v7::Gather,
                               ov::op::v8::Gather);

pass::EliminateReduceReshape::EliminateReduceReshape() {
    MATCHER_SCOPE(EliminateReduceReshape);
    using namespace pass::pattern;
    auto axes = wrap_type<ov::op::v0::Constant>();
    auto reduce_pattern = wrap_type<ov::op::util::ReductionBase>({any_input(), axes});
    auto requested_shape_pattern = wrap_type<ov::op::v0::Constant>();
    auto reshape_pattern =
        wrap_type<ov::op::v1::Reshape>({reduce_pattern, requested_shape_pattern}, consumers_count(1));

    matcher_pass_callback callback = [=](Matcher& m) {
        auto pattern_map = m.get_pattern_map();
        auto reshape_node = m.get_match_root();
        auto reduce_node = pattern_map.at(reduce_pattern);

        auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(reshape_node);
        auto reduce = ov::as_type_ptr<ov::op::util::ReductionBase>(reduce_node);
        if (!reshape || !reduce) {
            return false;
        }

        auto in_rank = reshape->get_input_partial_shape(0).rank();
        auto out_rank = reshape->get_output_partial_shape(0).rank();
        if (in_rank.is_dynamic() || out_rank.is_dynamic() || in_rank.get_length() != out_rank.get_length()) {
            return false;
        }

        auto requested_shape = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(requested_shape_pattern));
        if (!requested_shape) {
            return false;
        }

        auto requested_shape_vec = requested_shape->cast_vector<int64_t>();
        auto axes = reduce->get_reduction_axes();

        int cnt_dyn = 0;
        for (size_t i = 0; i < requested_shape_vec.size(); ++i) {
            // if we use reshape special zero or this dim was reduced
            cnt_dyn += !((requested_shape_vec[i] == 0 && reshape->get_special_zero()) ||
                         (axes.count(i) && requested_shape_vec[i] == 1));
        }

        // if the number of dyn dims here is equal to 0 or 1, we can unambiguously define output shape
        if (cnt_dyn <= 1) {
            return replace_output_update_name(reshape->output(0), reshape->input_value(0));
        } else {
            return false;
        }
    };

    auto m = make_shared<Matcher>(reshape_pattern, matcher_name);
    this->register_matcher(m, callback);
}

pass::EliminatePad::EliminatePad() {
    MATCHER_SCOPE(EliminatePad);
    auto pad_node_pattern = pattern::wrap_type<op::util::PadBase>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto pad = m.get_match_root();

        auto pad_begin_const = ov::util::get_constant_from_source(pad->input_value(1));
        auto pad_end_const = ov::util::get_constant_from_source(pad->input_value(2));

        if (!pad_begin_const || !pad_end_const) {
            return false;
        }

        const auto pad_begin_value = pad_begin_const->cast_vector<int64_t>();
        const auto pad_end_value = pad_end_const->cast_vector<int64_t>();

        if (any_of(pad_begin_value.begin(),
                   pad_begin_value.end(),
                   [](int64_t value) {
                       return value != 0;
                   }) ||
            any_of(pad_end_value.begin(), pad_end_value.end(), [](int64_t value) {
                return value != 0;
            })) {
            return false;
        }

        return replace_output_update_name(pad->output(0), pad->input_value(0));
    };

    auto m = make_shared<pattern::Matcher>(pad_node_pattern, matcher_name);
    this->register_matcher(m, callback);
}

pass::EliminateConvert::EliminateConvert() {
    MATCHER_SCOPE(EliminateConvert);
    auto convert_pattern = pattern::wrap_type<ov::op::v0::Convert>();

    matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto convert = dynamic_pointer_cast<ov::op::v0::Convert>(m.get_match_root());
        if (!convert) {
            return false;
        }
        if (convert->get_input_element_type(0) == convert->get_element_type()) {
            return replace_output_update_name(convert->output(0), convert->input_value(0));
        }
        return false;
    };

    auto m = make_shared<pattern::Matcher>(convert_pattern, matcher_name);
    this->register_matcher(m, callback);
}

pass::EliminateConvertNonZero::EliminateConvertNonZero() {
    MATCHER_SCOPE(EliminateConvertNonZero);
    auto convert_pattern = pattern::wrap_type<ov::op::v0::Convert>(pattern::consumers_count(1));
    auto non_zero = pattern::wrap_type<ov::op::v3::NonZero>({convert_pattern});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_map();
        auto convert = pattern_map.at(convert_pattern);
        // remove convert
        convert->output(0).replace(convert->input_value(0));
        // to make this elimination recursive we register NonZero as a node which will be used to repeat matching
        register_new_node(m.get_match_root());
        return true;
    };

    auto m = make_shared<pattern::Matcher>(non_zero, matcher_name);
    this->register_matcher(m, callback);
}

pass::EliminateConcat::EliminateConcat() {
    MATCHER_SCOPE(EliminateConcat);
    auto convert_pattern = pattern::wrap_type<ov::op::v0::Concat>();

    matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto concat = m.get_match_root();
        if (concat->inputs().size() == 1) {
            return replace_output_update_name(concat->output(0), concat->input_value(0));
        }
        return false;
    };

    auto m = make_shared<pattern::Matcher>(convert_pattern, matcher_name);
    this->register_matcher(m, callback);
}

pass::EliminateSplit::EliminateSplit() {
    MATCHER_SCOPE(EliminateSplit);
    auto convert_pattern = pattern::wrap_type<ov::op::v1::Split>();

    matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto split = dynamic_pointer_cast<ov::op::v1::Split>(m.get_match_root());
        if (!split || split->get_num_splits() != 1) {
            return false;
        }
        return replace_output_update_name(split->output(0), split->input_value(0));
    };

    auto m = make_shared<pattern::Matcher>(convert_pattern, matcher_name);
    this->register_matcher(m, callback);
}

pass::EliminateUnsqueeze::EliminateUnsqueeze() {
    MATCHER_SCOPE(EliminateUnsqueeze);
    auto unsqueeze_pattern = pattern::wrap_type<ov::op::v0::Unsqueeze>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        return eliminate_unsqueeze(m.get_match_root());
    };
    auto m = make_shared<pattern::Matcher>(unsqueeze_pattern, matcher_name);
    this->register_matcher(m, callback);
}

pass::EliminateSqueeze::EliminateSqueeze() {
    MATCHER_SCOPE(EliminateSqueeze);
    auto squeeze_pattern = pattern::wrap_type<ov::op::v0::Squeeze>();

    matcher_pass_callback callback = [](pattern::Matcher& m) {
        const auto node = m.get_match_root();
        auto out_shape = node->get_output_partial_shape(0);
        // try to replace all unsqueeze/squeeze with reshape
        if (out_shape.rank().is_static() && out_shape.rank().get_length() != 0 && count_unknown_dims(out_shape) < 2) {
            return replace_squeeze_unsqueeze(node);
        }

        auto squeeze = ov::as_type_ptr<ov::op::v0::Squeeze>(node);
        if (squeeze == nullptr)
            return false;
        auto input = squeeze->input_value(0).get_node_shared_ptr();
        auto replace_squeeze_only = [&](const vector<int64_t>& axes) {
            auto axes_const = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{axes.size()}, axes);
            auto new_sq = make_shared<ov::op::v0::Squeeze>(input->input_value(0), axes_const);
            if (squeeze->get_output_partial_shape(0).same_scheme(new_sq->get_output_partial_shape(0))) {
                return replace_node_update_name(squeeze, new_sq);
            }
            return false;
        };
        // eliminate redundant unsqueeze->squeeze
        if (auto unsqueeze = ov::as_type_ptr<ov::op::v0::Unsqueeze>(input)) {
            PartialShape data_shape;
            if (op::util::is_parameter(input)) {
                data_shape = unsqueeze->input(0).get_partial_shape();
            } else {
                data_shape = input->input(0).get_partial_shape();
            }
            if (ov::compare_constants(unsqueeze->input_value(1).get_node_shared_ptr(),
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
                    auto axes_const = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{axes.size()}, axes);
                    auto new_unsq = make_shared<ov::op::v0::Unsqueeze>(input->input_value(0), axes_const);
                    if (squeeze->get_output_partial_shape(0).same_scheme(new_unsq->get_output_partial_shape(0))) {
                        replace_output_update_name(squeeze, new_unsq);
                        return true;
                    }
                }
            }
            return false;
        }
        // eliminate redundant squeeze->squeeze
        if (auto squeeze_i = ov::as_type_ptr<ov::op::v0::Squeeze>(input)) {
            PartialShape data_shape;
            if (op::util::is_parameter(input)) {
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

    auto m = make_shared<pattern::Matcher>(squeeze_pattern, matcher_name);
    this->register_matcher(m, callback);
}

namespace {
int64_t make_positive(int64_t value, const Output<Node>& node) {
    const auto& rank = node.get_partial_shape().rank();
    if (value < 0 && rank.is_static()) {
        value = rank.get_length() + value;
    }
    return value;
};

bool check_squeeze(const shared_ptr<Node>& node) {
    auto squeeze = dynamic_pointer_cast<ov::op::v0::Squeeze>(node);
    if (squeeze) {
        auto axis = dynamic_pointer_cast<ov::op::v0::Constant>(squeeze->input_value(1).get_node_shared_ptr());
        if (axis) {
            auto axis_val = axis->cast_vector<int64_t>();
            if (axis_val.size() == 1 && make_positive(axis_val[0], squeeze->input_value(0)) == 1) {
                return true;
            }
        }
    }
    return false;
}

// Checks that Reshape actually equals to Squeeze op
// 0, -1 values in the shape pattern are not allowed.
bool check_reshape(const shared_ptr<Node>& node) {
    auto reshape = dynamic_pointer_cast<ov::op::v1::Reshape>(node);
    if (reshape) {
        auto shape_pattern = dynamic_pointer_cast<ov::op::v0::Constant>(reshape->input_value(1).get_node_shared_ptr());
        if (shape_pattern) {
            auto pattern_val = shape_pattern->cast_vector<int64_t>();
            bool is_valid_pattern = find(pattern_val.begin(), pattern_val.end(), 0) == pattern_val.end();
            is_valid_pattern =
                is_valid_pattern || find(pattern_val.begin(), pattern_val.end(), -1) == pattern_val.end();
            if (!is_valid_pattern) {
                return false;
            }
            pattern_val.insert(pattern_val.begin() + 1, 1);
            auto in_shape = reshape->input_value(0).get_partial_shape();
            // Current Reshape is a product of eliminate_reshape_v1 transformation.
            // Initial Unsqueeze operation had static input shape and thus was replaced.
            // This makes us eligible to assume input shape of Reshape that we are searching for is static
            if (in_shape.is_static() && in_shape == pattern_val) {
                return true;
            }
        }
    }
    return false;
}

bool check_axis(const shared_ptr<ov::op::v0::Concat>& concat,
                const shared_ptr<Node>& split,
                bool is_special_case = false) {
    auto axis = dynamic_pointer_cast<ov::op::v0::Constant>(split->input_value(1).get_node_shared_ptr());
    if (!axis) {
        return false;
    }
    const auto& axis_val = axis->cast_vector<int64_t>();
    if (axis_val.size() != 1 || (axis_val[0] != concat->get_axis() && make_positive(axis_val[0], split->output(0)) !=
                                                                          make_positive(concat->get_axis(), concat))) {
        return false;
    }

    // in case of LSTM/GRU/RNN Sequence case described below and Split/VariadicSplit op,
    // we have to check that the last slice length equals 1,
    // it corresponds output(1) of Seq op
    if (is_special_case) {
        auto last_out_shape = split->output(split->get_output_size() - 1).get_partial_shape();
        if (!last_out_shape.rank().is_static() || !last_out_shape[axis_val[0]].is_static() ||
            last_out_shape[axis_val[0]].get_length() != 1) {
            return false;
        }
    }
    return true;
}

template <class T>
shared_ptr<T> check_all_inputs(const shared_ptr<ov::op::v0::Concat>& concat) {
    shared_ptr<T> split;
    const auto concat_in_values = concat->input_values();
    size_t idx = 0;
    for (const auto& in_to_concat : concat_in_values) {
        const auto& cast_to_split = dynamic_pointer_cast<T>(in_to_concat.get_node_shared_ptr());
        // There is a special case with (GRU/RNN/LSTM)Sequence ops:
        //
        // (LSTM/GRU/RNN)Sequence -- output(0) --> Squeeze (Reshape) ->Split -(H1...Hn-1 outs) ->  Concat
        //                        -- output(1) Hn out ------------------------------------------>
        //
        // Sequence->output(0) is a concatenation of H1 ... Hn from each iteration
        // Sequence->output(1) is a Hn from the last iteration
        // where n is a number of iterations
        //
        // If we found Sequence->output(0) is split into separate H1...Hn but only H1...Hn-1 are used
        // for Concat and the last input to Concat is output(1) of Sequence op, which is actually Hn,
        // this is also a valid case for this Elimination.
        if (!cast_to_split) {
            if (idx != (concat_in_values.size() - 1) || !split) {
                return {};
            }
            shared_ptr<Node> in_to_split = split->input_value(0).get_node_shared_ptr();
            Output<Node> seq_out;
            if (in_to_split && !in_to_split->inputs().empty()) {
                seq_out = in_to_split->input_value(0);
            } else {
                return {};
            }

            auto seq_node = seq_out.get_node_shared_ptr();
            if (!seq_node || seq_out.get_index() != 0 ||
                !(dynamic_pointer_cast<ov::op::v5::RNNSequence>(seq_node) ||
                  dynamic_pointer_cast<ov::op::v5::GRUSequence>(seq_node) ||
                  dynamic_pointer_cast<ov::op::v5::LSTMSequence>(seq_node))) {
                return {};
            }

            // check that Split is connected to Sequence->output(0)
            // possible patterns:
            // Sequence:0->Squeeze->Split
            bool valid_pattern = check_squeeze(in_to_split);
            // Sequence:0->Reshape->Split
            if (!valid_pattern) {
                valid_pattern = check_reshape(in_to_split);
            }

            if (!valid_pattern) {
                return {};
            }

            // check that Sequence->output(1) is connected to this input or concat/split axis is not the same.
            if (!seq_node || in_to_concat != seq_node->output(1) || !check_axis(concat, split, true)) {
                return {};
            }
            return split;
        }
        // input (split op) should be the same for all inputs
        if (!split) {
            split = cast_to_split;
        } else if (cast_to_split.get() != split.get()) {
            // not all inputs to concat belong to the same Split op
            return {};
        }

        // Split to Concat edges are not in orderl
        // should be (0, 1, 2, ... , split->outputs().size()-1)
        if (in_to_concat.get_index() != idx) {
            return {};
        }
        ++idx;
    }

    // not all split outputs are used or concat/split axis is not the same.
    if (idx != split->outputs().size() || !check_axis(concat, split)) {
        return {};
    }

    return split;
}
}  // namespace

ov::pass::EliminateSplitConcat::EliminateSplitConcat() {
    MATCHER_SCOPE(EliminateSplitConcat);

    auto pattern_concat = pattern::wrap_type<ov::op::v0::Concat>();
    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_map();
        const auto concat = dynamic_pointer_cast<ov::op::v0::Concat>(pattern_map.at(pattern_concat));
        if (!concat) {
            return false;
        }
        shared_ptr<Node> split = check_all_inputs<ov::op::v1::Split>(concat);
        if (!split) {
            split = check_all_inputs<ov::op::v1::VariadicSplit>(concat);
        }

        if (!split) {
            return false;
        }

        return replace_output_update_name(concat->output(0), split->input_value(0));
    };

    auto m = make_shared<pattern::Matcher>(pattern_concat, matcher_name);
    this->register_matcher(m, callback);
}

pass::EliminateTranspose::EliminateTranspose() {
    MATCHER_SCOPE(EliminateTranspose);
    auto order = pattern::wrap_type<ov::op::v0::Constant>();
    auto transpose_pattern = pattern::wrap_type<ov::op::v1::Transpose>({pattern::any_input(), order});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_map();
        auto order_const = dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(order));
        if (!order_const) {
            return false;
        }

        const auto& order_values = order_const->cast_vector<int64_t>();
        if (order_values.empty()) {
            return false;
        }

        vector<int64_t> ref_values(order_values.size());
        iota(ref_values.begin(), ref_values.end(), 0);
        if (order_values != ref_values) {
            return false;
        }

        auto transpose = m.get_match_root();
        return replace_output_update_name(transpose->output(0), transpose->input_value(0));
    };

    auto m = make_shared<pattern::Matcher>(transpose_pattern, matcher_name);
    this->register_matcher(m, callback);
}

pass::EliminateEltwise::EliminateEltwise() {
    MATCHER_SCOPE(EliminateEltwise);
    auto input = pattern::any_input();
    auto constant_pattern = pattern::wrap_type<ov::op::v0::Constant>();
    auto eltwise_pattern =
        pattern::wrap_type<ov::op::v1::Add, ov::op::v1::Subtract, ov::op::v1::Multiply, ov::op::v1::Divide>(
            {input, constant_pattern});
    auto subtract_pattern =
        pattern::wrap_type<ov::op::v1::Subtract>({input, pattern::wrap_type<ov::op::v0::Convert>({constant_pattern})});
    auto root = make_shared<pattern::op::Or>(OutputVector{eltwise_pattern, subtract_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto eltwise = m.get_match_root();
        const auto& non_const_input = pattern_map.at(input);
        const auto& constant = pattern_map.at(constant_pattern);

        if (!op::util::can_eliminate_eltwise_node(eltwise, constant, non_const_input)) {
            return false;
        }
        return replace_output_update_name(eltwise->output(0), non_const_input);
    };

    auto m = make_shared<pattern::Matcher>(root, matcher_name);
    this->register_matcher(m, callback);
}

pass::EliminateScatterUpdate::EliminateScatterUpdate() {
    MATCHER_SCOPE(EliminateScatterUpdate);
    auto scatter_pattern = pattern::wrap_type<ov::op::v3::ScatterUpdate,
                                              ov::op::v3::ScatterNDUpdate,
                                              ov::op::v15::ScatterNDUpdate,
                                              ov::op::v3::ScatterElementsUpdate>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto scatter = m.get_match_root();
        const auto& indices_pshape = scatter->get_input_partial_shape(1);
        const auto& updates_pshape = scatter->get_input_partial_shape(2);

        auto has_zero = [](const ov::PartialShape& shape) -> bool {
            return std::any_of(shape.cbegin(), shape.cend(), ov::cmp::Equal<ov::Dimension>(0));
        };
        if (has_zero(indices_pshape) || has_zero(updates_pshape)) {
            return replace_output_update_name(scatter->output(0), scatter->input_value(0));
        } else {
            return false;
        }
    };

    auto m = make_shared<pattern::Matcher>(scatter_pattern, matcher_name);
    this->register_matcher(m, callback);
}

ov::pass::EliminateNopBroadcast::EliminateNopBroadcast() {
    MATCHER_SCOPE(EliminateNopBroadcast);
    auto root = pattern::wrap_type<op::v1::Broadcast, op::v3::Broadcast, op::v0::Tile>(
        pattern::op::as_value_predicate([](std::shared_ptr<Node> node) {
            auto input_rank = node->get_input_partial_shape(0).rank();
            auto output_rank = node->get_output_partial_shape(0).rank();
            return input_rank.is_static() && output_rank.is_static() && input_rank == output_rank;
        }));

    ov::matcher_pass_callback matcher_pass_callback = [](pattern::Matcher& m) {
        const auto& op = m.get_match_root();
        if (op::util::is_constant_and_all_values_equal_int(op->input_value(1), 1))
            return replace_output_update_name(op->output(0), op->input_value(0));
        return false;
    };

    auto m = std::make_shared<pattern::Matcher>(root, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ov::pass::EliminateSliceBeforeGatherElements::EliminateSliceBeforeGatherElements() {
    MATCHER_SCOPE(EliminateSliceBeforeGatherElements);
    auto slice = pattern::wrap_type<op::v8::Slice>();
    auto gather = pattern::wrap_type<op::v6::GatherElements>({slice, pattern::any_input()});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_node = m.get_pattern_map();
        const auto& slice_node = pattern_to_node.at(slice);
        bool start_from_zero = op::util::is_constant_and_all_values_equal_int(slice_node->input_value(1), 0);
        bool step_is_one = op::util::is_constant_and_all_values_equal_int(slice_node->input_value(3), 1);
        if (!start_from_zero || !step_is_one)
            return false;
        const auto& gather_node = pattern_to_node.at(gather);
        gather_node->input(0).replace_source_output(slice_node->input_value(0));
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(gather, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
ov::pass::EliminateSlice::EliminateSlice() {
    MATCHER_SCOPE(EliminateSlice);

    auto input = pattern::any_input();
    auto begin_const = pattern::wrap_type<ov::op::v0::Constant>();
    auto end_const = pattern::wrap_type<ov::op::v0::Constant>();
    auto step_const = pattern::wrap_type<ov::op::v0::Constant>();
    auto axes = pattern::any_input();
    auto pattern = pattern::wrap_type<ov::op::v8::Slice>({input, begin_const, end_const, step_const, axes});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        auto slice = ov::as_type_ptr<ov::op::v8::Slice>(m.get_match_root());
        if (!slice) {
            return false;
        }

        int64_t max_int = slice->input_value(2).get_element_type() == element::i32
                              ? std::numeric_limits<int32_t>::max()
                              : std::numeric_limits<int64_t>::max();
        bool is_nop = op::util::is_constant_and_all_values_equal_int(slice->input_value(1), 0) &&
                      op::util::is_constant_and_all_values_equal_int(slice->input_value(2), max_int) &&
                      op::util::is_constant_and_all_values_equal_int(slice->input_value(3), 1);

        if (is_nop) {
            return replace_output_update_name(slice->output(0), slice->input_value(0));
        } else {
            return false;
        }
    };
    auto m = std::make_shared<pattern::Matcher>(pattern, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ov::pass::EliminateStridedSlice::EliminateStridedSlice() {
    MATCHER_SCOPE(EliminateStridedSlice);

    auto input = pattern::any_input();
    auto begin_const = pattern::wrap_type<ov::op::v0::Constant>();
    auto end_const = pattern::wrap_type<ov::op::v0::Constant>();
    auto optional_stride_const = pattern::wrap_type<ov::op::v0::Constant>();
    auto pattern = pattern::wrap_type<ov::op::v1::StridedSlice>({input, begin_const, end_const, optional_stride_const});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        auto strided_slice_node = ov::as_type_ptr<ov::op::v1::StridedSlice>(m.get_match_root());
        if (!strided_slice_node) {
            return false;
        }
        // check that all values of the mask is equal 0
        auto check_mask = [](const std::vector<int64_t>& mask_to_check) {
            auto it = std::find_if(mask_to_check.begin(), mask_to_check.end(), [](const int64_t& value) {
                return value != 0;
            });
            if (mask_to_check.empty() || it == mask_to_check.end()) {
                return true;
            }
            return false;
        };
        // check that we won't do change dimention rank
        if (!check_mask(strided_slice_node->get_shrink_axis_mask()) ||
            !check_mask(strided_slice_node->get_new_axis_mask()) ||
            !check_mask(strided_slice_node->get_ellipsis_mask())) {
            return false;
        }
        // check that that we will take all values
        if (strided_slice_node->get_input_size() == 4 &&
            !op::util::is_constant_and_all_values_equal_int(strided_slice_node->input_value(3), 1)) {
            return false;
        }

        auto align_vectors = [](std::vector<int64_t>& vec_1, std::vector<int64_t>& vec_2) {
            auto max_size = std::max(vec_1.size(), vec_2.size());
            while (vec_1.size() < max_size) {
                vec_1.push_back(0);
            }
            while (vec_2.size() < max_size) {
                vec_2.push_back(0);
            }
            return;
        };
        auto begin_node = strided_slice_node->get_input_node_shared_ptr(1);
        if (const auto& begin_constant_node = ov::util::get_constant_from_source(begin_node)) {
            auto values = begin_constant_node->cast_vector<int64_t>();
            auto begin_mask = strided_slice_node->get_begin_mask();
            // align begin_mask and values_vec by length
            align_vectors(values, begin_mask);
            for (size_t i = 0; i < begin_mask.size(); ++i) {
                // if mask == 1 then ignore the begin_mask_value else check
                // if values[i] == 0 then take whole tensor else take part of a tensor
                if (!begin_mask[i] && values[i]) {
                    return false;
                }
            }
        } else {
            return false;
        }

        auto end_node = strided_slice_node->get_input_node_shared_ptr(2);
        if (const auto& end_constant_node = ov::util::get_constant_from_source(end_node)) {
            int64_t max_value = end_node->get_element_type() == ov::element::i32 ? std::numeric_limits<int32_t>::max()
                                                                                 : std::numeric_limits<int64_t>::max();

            auto values = end_constant_node->cast_vector<int64_t>();
            auto end_mask = strided_slice_node->get_end_mask();
            // align end_mask and values_vec by length
            align_vectors(values, end_mask);
            for (size_t i = 0; i < end_mask.size(); ++i) {
                // if mask == 1 then ignore the begin_mask_value else check
                // if values[i] == max then take whole tensor else take part of a tensor
                if (!end_mask[i] && values[i] != max_value) {
                    return false;
                }
            }
        } else {
            return false;
        }
        return replace_output_update_name(strided_slice_node->output(0), strided_slice_node->input_value(0));
    };
    auto m = std::make_shared<pattern::Matcher>(pattern, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ov::pass::EliminateStridedSliceByShape::EliminateStridedSliceByShape() {
    MATCHER_SCOPE(EliminateStridedSliceByShape);

    auto input = pattern::any_input();
    auto begin = pattern::any_input();
    auto end = pattern::any_input();
    auto optional_stride_const = pattern::wrap_type<ov::op::v0::Constant>();
    auto strided_slice = pattern::wrap_type<ov::op::v1::StridedSlice>({input, begin, end, optional_stride_const});

    auto axes = pattern::any_input();
    auto slice = pattern::wrap_type<ov::op::v8::Slice>({input, begin, end, optional_stride_const, axes});
    auto pattern = std::make_shared<pattern::op::Or>(OutputVector{strided_slice, slice});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        auto node = m.get_match_root();
        if (node == nullptr) {
            return false;
        }
        auto strided_slice_node = ov::as_type_ptr<ov::op::v1::StridedSlice>(node);
        if (strided_slice_node) {
            // check that all values of the mask is equal 0
            auto check_mask = [](const std::vector<int64_t>& mask_to_check) {
                auto it = std::find_if(mask_to_check.begin(), mask_to_check.end(), [](const int64_t& value) {
                    return value != 0;
                });
                if (mask_to_check.empty() || it == mask_to_check.end()) {
                    return true;
                }
                return false;
            };
            // check that we won't do change dimention rank
            if (!check_mask(strided_slice_node->get_shrink_axis_mask()) ||
                !check_mask(strided_slice_node->get_new_axis_mask()) ||
                !check_mask(strided_slice_node->get_ellipsis_mask())) {
                return false;
            }
        }

        // check that that we will take all values
        if (node->get_input_size() >= 4 && !op::util::is_constant_and_all_values_equal_int(node->input_value(3), 1)) {
            return false;
        }

        if (node->get_input_partial_shape(0).is_static() && node->get_output_partial_shape(0).is_static()) {
            if (node->get_input_shape(0) == node->get_output_shape(0)) {
                return replace_output_update_name(node->output(0), node->input_value(0));
            }
        }
        return false;
    };

    auto m = std::make_shared<pattern::Matcher>(pattern, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ov::pass::PrepareShapeOpsForEliminationAroundBE::PrepareShapeOpsForEliminationAroundBE() {
    MATCHER_SCOPE(PrepareShapeOpsForEliminationAroundBE);
    auto first_label = pattern::wrap_type<op::v1::Reshape, op::v0::Squeeze, op::v1::StridedSlice, op::util::GatherBase>(
        pattern::rank_equals(0));
    auto other_input_label = pattern::any_input(pattern::rank_equals(0));
    auto binary_op_label = pattern::wrap_type<op::util::BinaryElementwiseArithmetic,
                                              op::util::BinaryElementwiseComparison,
                                              op::util::BinaryElementwiseLogical>({first_label, other_input_label},
                                                                                  pattern::consumers_count(1));
    auto second_label = pattern::wrap_type<op::v1::Reshape, op::v0::Unsqueeze>({binary_op_label, pattern::any_input()},
                                                                               pattern::rank_equals(1));

    ov::matcher_pass_callback matcher_pass_callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_to_node = m.get_pattern_map();

        auto second_node = pattern_to_node.at(second_label);
        auto binary = pattern_to_node.at(binary_op_label);

        auto lhs_node =
            ov::op::util::clone_try_fold(second_node, {binary->input_value(0), second_node->input_value(1)});
        auto rhs_node =
            ov::op::util::clone_try_fold(second_node, {binary->input_value(1), second_node->input_value(1)});

        register_new_node(lhs_node);
        register_new_node(rhs_node);

        binary->input(0).replace_source_output(lhs_node->output(0));
        binary->input(1).replace_source_output(rhs_node->output(0));
        binary->validate_and_infer_types();

        ov::copy_runtime_info(second_node, {lhs_node, rhs_node});

        replace_output_update_name(second_node->output(0), binary->output(0));
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(second_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

ov::pass::NopElimination::NopElimination(bool use_shape_for_elimination) {
    // shape-agnostic transformations
    ADD_MATCHER_FOR_THIS(EliminatePad)
    ADD_MATCHER_FOR_THIS(EliminateReduceReshape)
    ADD_MATCHER_FOR_THIS(EliminateConvert)
    ADD_MATCHER_FOR_THIS(EliminateConvertNonZero)
    ADD_MATCHER_FOR_THIS(EliminateConcat)
    ADD_MATCHER_FOR_THIS(EliminateSplit)
    ADD_MATCHER_FOR_THIS(EliminateTranspose)
    ADD_MATCHER_FOR_THIS(EliminateEltwise)
    ADD_MATCHER_FOR_THIS(EliminateSplitConcat)
    ADD_MATCHER_FOR_THIS(EliminateStridedSlice)
    ADD_MATCHER_FOR_THIS(EliminateSlice)

    // shape-dependent transformations
    if (use_shape_for_elimination) {
        ADD_MATCHER_FOR_THIS(EliminateScatterUpdate)
        ADD_MATCHER_FOR_THIS(EliminateReshape)
        ADD_MATCHER_FOR_THIS(EliminateSqueeze)
        ADD_MATCHER_FOR_THIS(EliminateUnsqueeze)
        ADD_MATCHER_FOR_THIS(PrepareShapeOpsForEliminationAroundBE)
        ADD_MATCHER_FOR_THIS(EliminateBroadcast)
        ADD_MATCHER_FOR_THIS(EliminateNopBroadcast)
        ADD_MATCHER_FOR_THIS(EliminateSliceBeforeGatherElements)
        ADD_MATCHER_FOR_THIS(EliminateStridedSliceByShape)
        ADD_MATCHER_FOR_THIS(EliminateGather)
    }
}
