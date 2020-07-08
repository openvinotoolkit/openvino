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
#include "ngraph/builder/split.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/split.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

void pass::ConstantFolding::construct_constant_split()
{
    auto data_label = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto axis_label =
        make_shared<pattern::op::Label>(element::i64, Shape{}, pattern::has_class<op::Constant>());
    auto split_pattern = make_shared<op::v1::Split>(data_label, axis_label, 0);

    auto constant_split_callback = [this, data_label, axis_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_split_callback against node = "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        const auto data_node = static_pointer_cast<op::Constant>(pattern_map[data_label]);
        const auto axis_node = static_pointer_cast<op::Constant>(pattern_map[axis_label]);
        const auto split = static_pointer_cast<op::v1::Split>(m.get_match_root());

        const auto axis_val = axis_node->cast_vector<int64_t>()[0];
        const auto norm_axis_val = ngraph::normalize_axis(
            split.get(), axis_val, data_node->get_output_partial_shape(0).rank());
        const auto slices = builder::split(data_node, split->get_num_splits(), norm_axis_val);

        int index = 0;
        for (auto& output : split->outputs())
        {
            output.replace(slices[index++]);
        }
        split->outputs().clear();
        construct_constant_slice();

        return true;
    };
    auto split_matcher =
        make_shared<pattern::Matcher>(split_pattern, "ConstantFolding.ConstantSplit");
    this->add_matcher(split_matcher, constant_split_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
