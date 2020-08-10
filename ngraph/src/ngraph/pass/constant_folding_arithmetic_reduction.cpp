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
#include "ngraph/op/constant.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/reduce_mean.hpp"
#include "ngraph/op/reduce_prod.hpp"
#include "ngraph/op/reduce_sum.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/runtime/reference/max.hpp"
#include "ngraph/runtime/reference/mean.hpp"
#include "ngraph/runtime/reference/min.hpp"
#include "ngraph/runtime/reference/product.hpp"
#include "ngraph/runtime/reference/sum.hpp"

using namespace std;
using namespace ngraph;

template <typename T>
static shared_ptr<op::Constant>
    fold_constant_arithmetic_reduction_helper(shared_ptr<op::Constant> constant,
                                              shared_ptr<Node> reduction_node)
{
    const Shape& out_shape = reduction_node->get_shape();
    runtime::AlignedBuffer buffer(shape_size(out_shape) * sizeof(T));
    T* data_ptr = buffer.get_ptr<T>();

    if (auto max = as_type_ptr<op::Max>(reduction_node))
    {
        runtime::reference::max<T>(constant->get_data_ptr<T>(),
                                   data_ptr,
                                   constant->get_output_shape(0),
                                   max->get_reduction_axes());
    }
    else if (auto reduce_max = as_type_ptr<op::v1::ReduceMax>(reduction_node))
    {
        runtime::reference::max<T>(constant->get_data_ptr<T>(),
                                   data_ptr,
                                   constant->get_output_shape(0),
                                   reduce_max->get_reduction_axes());
    }
    else if (auto min = as_type_ptr<op::Min>(reduction_node))
    {
        runtime::reference::min<T>(constant->get_data_ptr<T>(),
                                   data_ptr,
                                   constant->get_output_shape(0),
                                   min->get_reduction_axes());
    }
    else if (auto reduce_min = as_type_ptr<op::v1::ReduceMin>(reduction_node))
    {
        runtime::reference::min<T>(constant->get_data_ptr<T>(),
                                   data_ptr,
                                   constant->get_output_shape(0),
                                   reduce_min->get_reduction_axes());
    }
    else if (auto prod = as_type_ptr<op::Product>(reduction_node))
    {
        runtime::reference::product<T>(constant->get_data_ptr<T>(),
                                       data_ptr,
                                       constant->get_output_shape(0),
                                       prod->get_reduction_axes());
    }
    else if (auto reduce_prod = as_type_ptr<op::v1::ReduceProd>(reduction_node))
    {
        runtime::reference::product<T>(constant->get_data_ptr<T>(),
                                       data_ptr,
                                       constant->get_output_shape(0),
                                       reduce_prod->get_reduction_axes());
    }
    else if (auto sum = as_type_ptr<op::Sum>(reduction_node))
    {
        runtime::reference::sum<T>(constant->get_data_ptr<T>(),
                                   data_ptr,
                                   constant->get_output_shape(0),
                                   sum->get_reduction_axes());
    }
    else if (auto reduce_sum = as_type_ptr<op::v1::ReduceSum>(reduction_node))
    {
        runtime::reference::sum<T>(constant->get_data_ptr<T>(),
                                   data_ptr,
                                   constant->get_output_shape(0),
                                   reduce_sum->get_reduction_axes());
    }
    else if (auto reduce_mean = as_type_ptr<op::v1::ReduceMean>(reduction_node))
    {
        runtime::reference::mean<T>(constant->get_data_ptr<T>(),
                                    data_ptr,
                                    constant->get_output_shape(0),
                                    reduce_mean->get_reduction_axes());
    }
    else
    {
        NGRAPH_CHECK(false,
                     "Internal nGraph error: Ops handled in "
                     "fold_constant_arithmetic_reduction_helper must be consistent with those "
                     "matched in construct_constant_arithmetic_reduction");
    }

    return make_shared<op::Constant>(
        reduction_node->get_output_element_type(0), reduction_node->get_shape(), data_ptr);
}

static shared_ptr<op::Constant>
    fold_constant_arithmetic_reduction(shared_ptr<op::Constant> constant,
                                       shared_ptr<Node> reduction_node)
{
    auto& input_element_type = constant->get_output_element_type(0);

    switch (input_element_type)
    {
    case element::Type_t::undefined:
        NGRAPH_CHECK(false,
                     "Encountered 'undefined' element type in fold_constant_arithmetic_reduction");
        break;
    case element::Type_t::dynamic:
        NGRAPH_CHECK(false,
                     "Encountered 'dynamic' element type in fold_constant_arithmetic_reduction");
        break;
    case element::Type_t::u1:
        NGRAPH_CHECK(false, "Encountered 'u1' element type in fold_constant_arithmetic_reduction");
        break;
    case element::Type_t::boolean:
        return fold_constant_arithmetic_reduction_helper<char>(constant, reduction_node);
    case element::Type_t::bf16:
        return fold_constant_arithmetic_reduction_helper<bfloat16>(constant, reduction_node);
    case element::Type_t::f16:
        return fold_constant_arithmetic_reduction_helper<float16>(constant, reduction_node);
    case element::Type_t::f32:
        return fold_constant_arithmetic_reduction_helper<float>(constant, reduction_node);
    case element::Type_t::f64:
        return fold_constant_arithmetic_reduction_helper<double>(constant, reduction_node);
    case element::Type_t::i8:
        return fold_constant_arithmetic_reduction_helper<int8_t>(constant, reduction_node);
    case element::Type_t::i16:
        return fold_constant_arithmetic_reduction_helper<int16_t>(constant, reduction_node);
    case element::Type_t::i32:
        return fold_constant_arithmetic_reduction_helper<int32_t>(constant, reduction_node);
    case element::Type_t::i64:
        return fold_constant_arithmetic_reduction_helper<int64_t>(constant, reduction_node);
    case element::Type_t::u8:
        return fold_constant_arithmetic_reduction_helper<uint8_t>(constant, reduction_node);
    case element::Type_t::u16:
        return fold_constant_arithmetic_reduction_helper<uint16_t>(constant, reduction_node);
    case element::Type_t::u32:
        return fold_constant_arithmetic_reduction_helper<uint32_t>(constant, reduction_node);
    case element::Type_t::u64:
        return fold_constant_arithmetic_reduction_helper<uint64_t>(constant, reduction_node);
    }

    NGRAPH_UNREACHABLE("Unexpected switch case");
}

void pass::ConstantFolding::construct_constant_arithmetic_reduction()
{
    auto constant_data_label = make_shared<pattern::op::Label>(
        element::i32, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto constant_axes_label =
        make_shared<pattern::op::Label>(element::i64, Shape{2}, pattern::has_class<op::Constant>());
    auto is_supported_reduction = [](std::shared_ptr<Node> n) {
        return (pattern::has_class<op::Max>()(n) || pattern::has_class<op::Min>()(n) ||
                pattern::has_class<op::Product>()(n) || pattern::has_class<op::Sum>()(n) ||
                pattern::has_class<op::v1::ReduceMax>()(n) ||
                pattern::has_class<op::v1::ReduceMin>()(n) ||
                pattern::has_class<op::v1::ReduceProd>()(n) ||
                pattern::has_class<op::v1::ReduceSum>()(n) ||
                pattern::has_class<op::v1::ReduceMean>()(n));
    };
    auto reduction =
        std::make_shared<pattern::op::Any>(element::i32,
                                           Shape{2},
                                           is_supported_reduction,
                                           NodeVector{constant_data_label, constant_axes_label});

    auto constant_arithmetic_reduction_callback = [constant_data_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_arithmetic_reduction_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = static_pointer_cast<op::Constant>(pattern_map[constant_data_label]);
        auto reduction_match = m.get_match_root();

        NGRAPH_CHECK(revalidate_and_ensure_static(reduction_match));

        replace_node(reduction_match,
                     fold_constant_arithmetic_reduction(constant_match, reduction_match));
        return true;
    };

    auto arithmetic_reduction_matcher =
        make_shared<pattern::Matcher>(reduction, "ConstantFolding.ConstantArithmeticReduction");
    NGRAPH_SUPPRESS_DEPRECATED_START
    this->add_matcher(arithmetic_reduction_matcher,
                      constant_arithmetic_reduction_callback,
                      PassProperty::CHANGE_DYNAMIC_STATE);
    NGRAPH_SUPPRESS_DEPRECATED_END
}
