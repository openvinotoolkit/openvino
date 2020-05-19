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

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            class NGRAPH_API CropAndResize : public Op
            {
            public:
                enum class ResizeMethod
                {
                    unspecified,
                    bilinear,
                    nearest
                };

                static constexpr NodeTypeInfo type_info{"CropAndResize", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a crop and resize operation.
                CropAndResize() = default;

                /// \param image [N, H, W, C]
                /// \param boxes [NUM_BOXES, 4] where boxes[box] is [y1, x1, y2, x2] each in [0, 1]
                /// \param box_indices [NUM_BOXES] in [0, N)
                /// \param crop_size [crop_height, crop_width]
                CropAndResize(const Output<Node>& image,
                              const Output<Node>& boxes,
                              const Output<Node>& box_indices,
                              const Output<Node>& crop_size,
                              ResizeMethod resize_method,
                              float extrapolation_value);

                void validate_and_infer_types() override;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                ResizeMethod get_resize_method() const { return m_resize_method; }
                void set_resize_method(ResizeMethod resize_method)
                {
                    m_resize_method = resize_method;
                }
                float get_extrapolation_value() const { return m_extrapolation_value; }
                void set_extrapolation_value(float extrapolation_value)
                {
                    m_extrapolation_value = extrapolation_value;
                }

            private:
                ResizeMethod m_resize_method{ResizeMethod::unspecified};
                float m_extrapolation_value{0};
            };
        }
        using v0::CropAndResize;
    }

    const std::string& as_string(op::CropAndResize::ResizeMethod);
    template <typename T>
    T as_type(const std::string&);

    template <>
    op::CropAndResize::ResizeMethod as_type<op::CropAndResize::ResizeMethod>(const std::string&);
}
