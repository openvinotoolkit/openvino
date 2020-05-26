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

#include <vector>

#include "ngraph/op/constant.hpp"
#include "ngraph/op/crop_and_resize.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::CropAndResize::type_info;

op::CropAndResize::CropAndResize(const Output<Node>& image,
                                 const Output<Node>& boxes,
                                 const Output<Node>& box_indices,
                                 const Output<Node>& crop_size,
                                 ResizeMethod resize_method,
                                 float extrapolation_value)
    : Op({image, boxes, box_indices, crop_size})
    , m_resize_method(resize_method)
    , m_extrapolation_value(extrapolation_value)
{
    constructor_validate_and_infer_types();
}

void op::CropAndResize::validate_and_infer_types()
{
    NODE_VALIDATION_CHECK(this, get_input_size() == 4);
    NODE_VALIDATION_CHECK(
        this, m_resize_method != ResizeMethod::unspecified, "Resize method not specified");
    auto image = input_value(0);
    auto& image_et = image.get_element_type();

    // Will override if we can determine the shape
    set_output_type(0, image_et, {});

    auto image_shape = image.get_partial_shape();
    Dimension image_depth;
    if (image_shape.is_static())
    {
        NODE_VALIDATION_CHECK(this, image_shape.rank().get_length() == 4, "Image must be NHWC");
        image_depth = image_shape[3];
    }

    auto boxes = input_value(1);
    auto boxes_shape = boxes.get_partial_shape();
    if (boxes_shape.is_static())
    {
        auto boxes_rank = boxes_shape.rank();
        NODE_VALIDATION_CHECK(this, boxes_rank.get_length() == 2, "Boxes must be 2d");
        auto boxes_dim1 = boxes_shape[1];
        NODE_VALIDATION_CHECK(
            this, boxes_dim1.get_length() == 4, "Second boxes dimension must be 4");
    }
    NODE_VALIDATION_CHECK(
        this, boxes.get_element_type().is_real(), "Boxes must be real values in [0, 1]");

    auto box_indices = input_value(2);
    auto box_indices_shape = box_indices.get_partial_shape();
    Dimension num_boxes;
    if (box_indices_shape.is_static())
    {
        NODE_VALIDATION_CHECK(
            this, box_indices_shape.rank().get_length() == 1, "Box indices must have rank 1");
        num_boxes = box_indices_shape[0];
    }
    NODE_VALIDATION_CHECK(
        this, box_indices.get_element_type().is_integral(), "Box indices must be integers");

    auto crop_size = input_value(3);
    auto crop_size_shape = crop_size.get_partial_shape();
    auto crop_size_rank = crop_size_shape.rank();
    NODE_VALIDATION_CHECK(this,
                          crop_size_shape.is_static() || crop_size_rank.is_dynamic(),
                          "Dynamic crop_size not supported");

    NODE_VALIDATION_CHECK(this, crop_size_rank.get_length() == 1, "crop_size must be a vector");
    NODE_VALIDATION_CHECK(
        this, crop_size_shape[0].get_length() == 2, "crop_size must be a vector of length 2");
    auto& crop_size_et = crop_size.get_element_type();
    NODE_VALIDATION_CHECK(this, crop_size_et.is_integral(), "crops_size must be integral");
    auto crop_size_node = crop_size.get_node_shared_ptr();
    NODE_VALIDATION_CHECK(this, is_type<op::v0::Constant>(crop_size_node), "crop_size must be a constant");
    auto crop_size_const = static_pointer_cast<op::Constant>(crop_size_node);
    if (crop_size_et == element::i8)
    {
        auto v = crop_size_const->get_vector<int8_t>();
        set_output_type(0, image_et, {num_boxes, v[0], v[1], image_depth});
    }
    else if (crop_size_et == element::u8)
    {
        auto v = crop_size_const->get_vector<uint8_t>();
        set_output_type(0, image_et, {num_boxes, v[0], v[1], image_depth});
    }
    else if (crop_size_et == element::i16)
    {
        auto v = crop_size_const->get_vector<int16_t>();
        set_output_type(0, image_et, {num_boxes, v[0], v[1], image_depth});
    }
    else if (crop_size_et == element::u16)
    {
        auto v = crop_size_const->get_vector<uint16_t>();
        set_output_type(0, image_et, {num_boxes, v[0], v[1], image_depth});
    }
    else if (crop_size_et == element::i32)
    {
        auto v = crop_size_const->get_vector<int32_t>();
        set_output_type(0, image_et, {num_boxes, v[0], v[1], image_depth});
    }
    else if (crop_size_et == element::u32)
    {
        auto v = crop_size_const->get_vector<uint32_t>();
        set_output_type(0, image_et, {num_boxes, v[0], v[1], image_depth});
    }
    else if (crop_size_et == element::i64)
    {
        auto v = crop_size_const->get_vector<int64_t>();
        set_output_type(0, image_et, {num_boxes, v[0], v[1], image_depth});
    }
    else if (crop_size_et == element::u64)
    {
        auto v = crop_size_const->get_vector<uint64_t>();
        set_output_type(
            0,
            image_et,
            {num_boxes, static_cast<int64_t>(v[0]), static_cast<int64_t>(v[1]), image_depth});
    }
    else
    {
        NODE_VALIDATION_CHECK(this, false, "Unknown integral type for crop size");
    }
}

shared_ptr<Node> op::CropAndResize::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<CropAndResize>(new_args.at(0),
                                      new_args.at(1),
                                      new_args.at(2),
                                      new_args.at(3),
                                      m_resize_method,
                                      m_extrapolation_value);
}

static const vector<pair<string, op::CropAndResize::ResizeMethod>>& get_resize_pairs()
{
    static vector<pair<string, op::CropAndResize::ResizeMethod>> pairs{
        {"unspecified", op::CropAndResize::ResizeMethod::unspecified},
        {"bilinear", op::CropAndResize::ResizeMethod::bilinear},
        {"nearest", op::CropAndResize::ResizeMethod::nearest}};
    return pairs;
}

const string& ngraph::as_string(op::CropAndResize::ResizeMethod resize_method)
{
    for (auto& p : get_resize_pairs())
    {
        if (p.second == resize_method)
        {
            return p.first;
        }
    }
    throw ngraph_error("Internal error: unhandled resize method");
}

namespace ngraph
{
    template <>
    op::CropAndResize::ResizeMethod as_type<op::CropAndResize::ResizeMethod>(const std::string& s)
    {
        for (auto& p : get_resize_pairs())
        {
            if (p.first == s)
            {
                return p.second;
            }
        }
        throw ngraph_error("Internal error: unhandled resize method name");
    }
}
