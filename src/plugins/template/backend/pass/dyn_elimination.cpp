// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dyn_elimination.hpp"

#include <numeric>

#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/range.hpp"
#include "ngraph/op/transpose.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/runtime/reference/range.hpp"
#include "ngraph/slice_plan.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

pass::DynElimination::DynElimination() : GraphRewrite() {
    construct_range();
}

template <typename T>
std::shared_ptr<op::Constant> make_range_replacement(const element::Type& et,
                                                     const Shape& shape,
                                                     const std::shared_ptr<op::Constant>& start_arg,
                                                     const std::shared_ptr<op::Constant>& step_arg) {
    std::vector<T> elements(shape_size(shape));
    std::vector<T> start_vec = start_arg->get_vector<T>();
    std::vector<T> step_vec = step_arg->get_vector<T>();

    NGRAPH_CHECK(start_vec.size() == 1 && step_vec.size() == 1);

    runtime::reference::range<T>(start_vec.data(), step_vec.data(), shape_size(shape), elements.data());

    return make_shared<op::Constant>(et, shape, elements);
}

void pass::DynElimination::construct_range() {
    auto start_arg_label = make_shared<pattern::op::Label>(element::f32, Shape{}, pattern::has_class<op::Constant>());
    auto stop_arg_label = make_shared<pattern::op::Label>(element::f32, Shape{}, pattern::has_class<op::Constant>());
    auto step_arg_label = make_shared<pattern::op::Label>(element::f32, Shape{}, pattern::has_class<op::Constant>());

    auto range_pat = make_shared<op::Range>(start_arg_label, stop_arg_label, step_arg_label);

    auto range_callback = [start_arg_label, stop_arg_label, step_arg_label](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_map();

        auto start_arg = static_pointer_cast<op::Constant>(pattern_map[start_arg_label]);
        auto step_arg = static_pointer_cast<op::Constant>(pattern_map[step_arg_label]);
        auto range_node = static_pointer_cast<op::Range>(m.get_match_root());

        NGRAPH_CHECK(start_arg->get_output_partial_shape(0).rank().compatible(0) &&
                     step_arg->get_output_partial_shape(0).rank().compatible(0));

        auto et = range_node->get_output_element_type(0);
        auto shape = range_node->get_output_shape(0);

        std::shared_ptr<op::Constant> replacement;

#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic error "-Wswitch"
#    pragma GCC diagnostic error "-Wswitch-enum"
#endif
        switch (et) {
        case element::Type_t::bf16:
            replacement = make_range_replacement<bfloat16>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::f16:
            replacement = make_range_replacement<float16>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::f32:
            replacement = make_range_replacement<float>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::f64:
            replacement = make_range_replacement<double>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::i8:
            replacement = make_range_replacement<int8_t>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::i16:
            replacement = make_range_replacement<int16_t>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::i32:
            replacement = make_range_replacement<int32_t>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::i64:
            replacement = make_range_replacement<int64_t>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::u8:
            replacement = make_range_replacement<uint8_t>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::u16:
            replacement = make_range_replacement<uint16_t>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::u32:
            replacement = make_range_replacement<uint32_t>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::u64:
            replacement = make_range_replacement<uint64_t>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::i4:
        case element::Type_t::u4:
        case element::Type_t::u1:
        case element::Type_t::undefined:
        case element::Type_t::dynamic:
        case element::Type_t::boolean:
            NGRAPH_CHECK(false, "Internal nGraph error: unsupported element type: ", et);
            break;
        }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#    pragma GCC diagnostic pop
#endif

        replace_node(range_node, replacement);
        return true;
    };

    auto range_matcher = make_shared<pattern::Matcher>(range_pat, "DynElimination.Range");
    NGRAPH_SUPPRESS_DEPRECATED_START
    add_matcher(range_matcher, range_callback, all_pass_property_off);
    NGRAPH_SUPPRESS_DEPRECATED_END
}
