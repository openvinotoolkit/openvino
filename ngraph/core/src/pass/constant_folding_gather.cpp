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

#include "constant_folding.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/gather.hpp"
#include "ngraph/op/squeeze.hpp"
#include "ngraph/runtime/reference/gather.hpp"

using namespace std;
using namespace ngraph;

void pass::ConstantFolding::construct_constant_gather_with_subgraph()
{
    auto concat_label = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 3, 4}, pattern::has_class<op::Concat>());
    auto indices_label =
        make_shared<pattern::op::Label>(element::i64, Shape{5}, pattern::has_class<op::Constant>());
    auto axis_label =
        make_shared<pattern::op::Label>(element::i64, Shape{1}, pattern::has_class<op::Constant>());
    auto gather_v1 = make_shared<op::v1::Gather>(concat_label, indices_label, axis_label);

    auto concat_gather_callback = [this, concat_label, indices_label, axis_label](
        pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for construct_constant_gather_with_subgraph against node = "
                     << m.get_match_root();

        auto pattern_map = m.get_pattern_map();

        const auto concat = static_pointer_cast<op::Concat>(pattern_map[concat_label]);

        const auto indices = static_pointer_cast<op::Constant>(pattern_map[indices_label]);
        const auto axis = static_pointer_cast<op::Constant>(pattern_map[axis_label]);
        const auto gather = m.get_match_root();

        if (cf_is_disabled(gather))
            return false;

        // only along axis=0
        if (axis->cast_vector<int64_t>()[0] != 0 || concat->get_axis() != 0)
            return false;
        // only single indices are accepted
        const auto indices_shape = indices->get_shape();
        if (indices_shape.size() > 1 || (indices_shape.size() == 1 && indices_shape[0] > 1))
            return false;
        // concat inputs are 1D and their count is equal to Concat output shape
        if (concat->get_output_partial_shape(0).is_dynamic())
            return false;
        const auto concat_inputs = concat->inputs();
        // concat inputs must be single elements
        if (concat_inputs.size() != shape_size(concat->get_shape()))
            return false;

        const int64_t rank = concat->get_shape()[0];
        const int64_t raw_index = indices->cast_vector<int64_t>()[0];
        const int64_t positive_index = raw_index < 0 ? rank + raw_index : raw_index;
        NGRAPH_CHECK(positive_index >= 0 && positive_index < rank);

        // gather takes exactly one element out of the Concat output
        const auto gathered_concat_input =
            concat_inputs[positive_index].get_source_output().get_node_shared_ptr();
        // Concat inputs are 1D, resulting tensor shape depends on Gather indices
        auto gathered = gathered_concat_input;
        if (indices_shape.empty())
        {
            // gathering a scalar
            auto axes = op::Constant::create(element::i64, Shape{1}, {0});
            gathered = make_shared<op::v0::Squeeze>(gathered_concat_input, axes);
        }
        replace_node(m.get_match_root(), gathered);
        return true;
    };

    auto gather_matcher_v1 = make_shared<pattern::Matcher>(
        gather_v1, "ConstantFolding.ConstantGatherV1WithDynamicSubgraph");
    NGRAPH_SUPPRESS_DEPRECATED_START
    this->add_matcher(
        gather_matcher_v1, concat_gather_callback, PassProperty::CHANGE_DYNAMIC_STATE);
    NGRAPH_SUPPRESS_DEPRECATED_END
}
