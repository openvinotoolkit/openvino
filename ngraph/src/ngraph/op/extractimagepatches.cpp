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

#include "ngraph/op/extractimagepatches.hpp"
#include "ngraph/attribute_visitor.hpp"

using namespace std;
using namespace ngraph;

// ExtractImagePatches v3

constexpr NodeTypeInfo op::v3::ExtractImagePatches::type_info;

op::v3::ExtractImagePatches::ExtractImagePatches(const Output<Node>& image,
                                                 const Shape& sizes,
                                                 const Strides& strides,
                                                 const Shape& rates,
                                                 const PadType& auto_pad)
    : Op({image})
    , m_patch_sizes(sizes)
    , m_patch_movement_strides(strides)
    , m_patch_selection_rates(rates)
    , m_padding(auto_pad)
{
    constructor_validate_and_infer_types();
}

void op::v3::ExtractImagePatches::validate_and_infer_types()
{
    const PartialShape input_Pshape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0).is_dynamic() ||
                              get_input_element_type(0).is_integral_number(),
                          "input tensor must be an integral number.");
    NODE_VALIDATION_CHECK(this, input_Pshape.rank() == 4, "input tensor must be 4D tensor.");

    NODE_VALIDATION_CHECK(this,
                          m_patch_sizes.size() == 2,
                          "Attribute sizes should be in [size_rows, size_cols] format.");

    NODE_VALIDATION_CHECK(this,
                          m_patch_movement_strides.size() == 2,
                          "Attribute strides should be in [stride_rows, stride_cols] format.");

    NODE_VALIDATION_CHECK(this,
                          m_patch_movement_strides[0] > 0 && m_patch_movement_strides[1] > 0,
                          "Attribute strides should be strictly greater than zeros in values.");

    NODE_VALIDATION_CHECK(this,
                          m_patch_selection_rates.size() == 2,
                          "Attribute rates should be in [rate_rows, rate_cols] format.");

    NODE_VALIDATION_CHECK(this,
                          m_patch_selection_rates[0] > 0 && m_patch_selection_rates[1] > 0,
                          "Attribute rates should be strictly greater than zeros in values.");

    NODE_VALIDATION_CHECK(
        this,
        m_padding == PadType::VALID || m_padding == PadType::SAME_LOWER ||
            m_padding == PadType::SAME_UPPER,
        "Attribute padding should be in either valid or same_lower or same_upper.");

    if (input_Pshape[1].is_dynamic() || input_Pshape[2].is_dynamic() ||
        input_Pshape[3].is_dynamic())
    {
        set_input_is_relevant_to_shape(0);
        auto output_Pshape = PartialShape::dynamic(4);
        set_output_type(0, get_input_element_type(0), output_Pshape);
    }
    else
    {
        int32_t input_depth = input_Pshape[1].get_length();
        int32_t input_rows = input_Pshape[2].get_length();
        int32_t input_cols = input_Pshape[3].get_length();
        int32_t out_rows(0);
        int32_t out_cols(0);

        if (input_rows == 0 || input_cols == 0)
        {
            out_rows = 0;
            out_cols = 0;
        }
        else if (m_padding == PadType::VALID)
        {
            out_rows = (((input_rows) -
                         static_cast<int32_t>(m_patch_selection_rates[0]) *
                             (static_cast<int32_t>(m_patch_sizes[0]) - 1) -
                         1) /
                        m_patch_movement_strides[0]) +
                       1;
            out_cols = (((input_cols) -
                         static_cast<int32_t>(m_patch_selection_rates[1]) *
                             (static_cast<int32_t>(m_patch_sizes[1]) - 1) -
                         1) /
                        m_patch_movement_strides[1]) +
                       1;
        }
        else
        {
            out_rows = 1 + (((input_rows)-1) / m_patch_movement_strides[0]);
            out_cols = 1 + (((input_cols)-1) / m_patch_movement_strides[1]);
        }

        if (out_rows < 0)
            out_rows = 0;
        if (out_cols < 0)
            out_cols = 0;

        ngraph::Dimension::value_type out_depth_cast = static_cast<ngraph::Dimension::value_type>(
            input_depth * m_patch_sizes[0] * m_patch_sizes[1]);
        ngraph::Dimension::value_type out_rows_cast =
            static_cast<ngraph::Dimension::value_type>(out_rows);
        ngraph::Dimension::value_type out_cols_cast =
            static_cast<ngraph::Dimension::value_type>(out_cols);

        PartialShape output_Pshape;
        if (input_Pshape[0].is_dynamic())
        {
            output_Pshape =
                PartialShape{input_Pshape[0], out_depth_cast, out_rows_cast, out_cols_cast};
        }
        else
        {
            ngraph::Dimension::value_type input_batch_cast =
                static_cast<ngraph::Dimension::value_type>(input_Pshape[0].get_length());
            output_Pshape =
                PartialShape{input_batch_cast, out_depth_cast, out_rows_cast, out_cols_cast};
        }

        if (input_rows == 0 || input_cols == 0)
        {
            output_Pshape = input_Pshape;
        }

        set_output_type(0, get_input_element_type(0), output_Pshape);
    }
}

bool op::v3::ExtractImagePatches::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("sizes", m_patch_sizes);
    visitor.on_attribute("strides", m_patch_movement_strides);
    visitor.on_attribute("rates", m_patch_selection_rates);
    visitor.on_attribute("auto_pad", m_padding);
    return true;
}

shared_ptr<Node>
    op::v3::ExtractImagePatches::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v3::ExtractImagePatches>(new_args.at(0),
                                                    m_patch_sizes,
                                                    m_patch_movement_strides,
                                                    m_patch_selection_rates,
                                                    m_padding);
}
