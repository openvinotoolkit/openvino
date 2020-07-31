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

#include "ngraph/op/ctc_loss.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v4::CTCLoss::type_info;

op::v4::CTCLoss::CTCLoss(const Output<Node>& logits,
                         const Output<Node>& logit_length,
                         const Output<Node>& labels,
                         const Output<Node>& label_length,
                         const bool preprocess_collapse_repeated,
                         const bool ctc_merge_repeated,
                         const bool unique)
    : Op({logits, logit_length, labels, label_length})
    , preprocess_collapse_repeated_(preprocess_collapse_repeated)
    , ctc_merge_repeated_(ctc_merge_repeated)
    , unique_(unique)
{
    constructor_validate_and_infer_types();
}

op::v4::CTCLoss::CTCLoss(const Output<Node>& logits,
                         const Output<Node>& logit_length,
                         const Output<Node>& labels,
                         const Output<Node>& label_length,
                         const Output<Node>& blank_index,
                         const bool preprocess_collapse_repeated,
                         const bool ctc_merge_repeated,
                         const bool unique)
    : Op({logits, logit_length, labels, label_length, blank_index})
    , preprocess_collapse_repeated_(preprocess_collapse_repeated)
    , ctc_merge_repeated_(ctc_merge_repeated)
    , unique_(unique)
{
    constructor_validate_and_infer_types();
}

void op::v4::CTCLoss::validate_and_infer_types()
{
    // check types of input tensors
    const auto& logits_type = get_input_element_type(0);
    const auto& logit_length_type = get_input_element_type(1);
    const auto& labels_type = get_input_element_type(2);
    const auto& label_length_type = get_input_element_type(3);

    NODE_VALIDATION_CHECK(this,
                          logits_type.is_real(),
                          "The data type for logits is expected to be a floating point type. Got: ",
                          logits_type);

    NODE_VALIDATION_CHECK(this,
                          logit_length_type.is_integral_number(),
                          "The logit length type is expected to be an integer type. Got: ",
                          logit_length_type);

    NODE_VALIDATION_CHECK(this,
                          labels_type.is_integral_number(),
                          "The labels type is expected to be an integer type. Got: ",
                          labels_type);

    NODE_VALIDATION_CHECK(this,
                          label_length_type.is_integral_number(),
                          "The label length type is expected to be an integer type. Got: ",
                          label_length_type);

    // check optional input type: blank index
    if (get_input_size() == 5)
    {
        const auto& blank_index_type = get_input_element_type(4);
        NODE_VALIDATION_CHECK(this,
                              blank_index_type.is_integral_number(),
                              "The blank index type is expected to be an integer type. Got: ",
                              blank_index_type);
    }

    // check ranks of input tensors
    const auto& logits_pshape = get_input_partial_shape(0);
    const auto& logit_length_pshape = get_input_partial_shape(1);
    const auto& labels_pshape = get_input_partial_shape(2);
    const auto& label_length_pshape = get_input_partial_shape(3);

    NODE_VALIDATION_CHECK(this,
                          logits_pshape.rank().compatible(3),
                          "Expected a 3D tensor for logits. Got: ",
                          logits_pshape);

    NODE_VALIDATION_CHECK(this,
                          logit_length_pshape.rank().compatible(1),
                          "Expected a 1D tensor for logit length. Got: ",
                          logit_length_pshape);

    NODE_VALIDATION_CHECK(this,
                          labels_pshape.rank().compatible(2),
                          "Expected a 2D tensor for labels. Got: ",
                          labels_pshape);

    NODE_VALIDATION_CHECK(this,
                          label_length_pshape.rank().compatible(1),
                          "Expected a 1D tensor for label length. Got: ",
                          label_length_pshape);

    // check optional input shape: blank index
    if (get_input_size() == 5)
    {
        const auto& blank_index_pshape = get_input_partial_shape(4);
        NODE_VALIDATION_CHECK(this,
                              blank_index_pshape.rank().compatible(0),
                              "Expected a scalar for blank index. Got: ",
                              blank_index_pshape);
    }

    // check shapes of input tensors
    size_t batch_size = 1;
    bool is_batch_size_set = false;
    size_t time_steps = 1;
    bool is_time_steps_set = false;

    if (logits_pshape.rank().is_static())
    {
        if (logits_pshape[0].is_static())
        {
            batch_size = logits_pshape[0].get_length();
            is_batch_size_set = true;
        }
        if (logits_pshape[1].is_static())
        {
            time_steps = logits_pshape[1].get_length();
            is_time_steps_set = true;
        }
    }

    if (logit_length_pshape.is_static())
    {
        if (is_batch_size_set)
        {
            NODE_VALIDATION_CHECK(
                this,
                logit_length_pshape[0].compatible(batch_size),
                "The first dimension of logit length must be equal to the first dimension ",
                "of the logits. Got: ",
                logit_length_pshape[0],
                " and: ",
                batch_size);
        }
        else if (logit_length_pshape[0].is_static())
        {
            batch_size = logit_length_pshape[0].get_length();
            is_batch_size_set = true;
        }
    }

    if (labels_pshape.is_static())
    {
        if (is_batch_size_set)
        {
            NODE_VALIDATION_CHECK(
                this,
                labels_pshape[0].compatible(batch_size),
                "The first dimension of labels must be equal to the first dimension ",
                "of the logits and the logit length. Got: ",
                labels_pshape[0],
                " and: ",
                batch_size);
        }
        else if (labels_pshape[0].is_static())
        {
            batch_size = labels_pshape[0].get_length();
            is_batch_size_set = true;
        }

        if (is_time_steps_set)
        {
            NODE_VALIDATION_CHECK(
                this,
                labels_pshape[1].compatible(time_steps),
                "The second dimension of labels must be equal to the second dimension ",
                "of logits. Got: ",
                labels_pshape[1],
                " and: ",
                time_steps);
        }
    }

    if (label_length_pshape.is_static())
    {
        if (!is_batch_size_set && label_length_pshape[0].is_static())
        {
            batch_size = label_length_pshape[0].get_length();
            is_batch_size_set = true;
        }
        NODE_VALIDATION_CHECK(
            this,
            label_length_pshape[0].compatible(batch_size),
            "The first dimension of label length must be equal to the first dimension ",
            "of the logits, the logit length and labels. Got: ",
            label_length_pshape[0],
            " and: ",
            batch_size);
    }

    // set output shape
    set_output_size(1);
    if (is_batch_size_set)
    {
        set_output_type(0, logits_type, Shape{batch_size});
    }
    else
    {
        set_output_type(0, logits_type, PartialShape{Dimension::dynamic()});
    }
}

bool op::v4::CTCLoss::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("preprocess_collapse_repeated", preprocess_collapse_repeated_);
    visitor.on_attribute("ctc_merge_repeated", ctc_merge_repeated_);
    visitor.on_attribute("unique", unique_);
    return true;
}

shared_ptr<Node> op::v4::CTCLoss::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    if (new_args.size() == 4)
    {
        return make_shared<CTCLoss>(new_args.at(0),
                                    new_args.at(1),
                                    new_args.at(2),
                                    new_args.at(3),
                                    preprocess_collapse_repeated_,
                                    ctc_merge_repeated_,
                                    unique_);
    }
    else if (new_args.size() == 5)
    {
        return make_shared<CTCLoss>(new_args.at(0),
                                    new_args.at(1),
                                    new_args.at(2),
                                    new_args.at(3),
                                    new_args.at(4),
                                    preprocess_collapse_repeated_,
                                    ctc_merge_repeated_,
                                    unique_);
    }
    else
    {
        throw ngraph_error("Incorrect number of arguments");
    }
}
