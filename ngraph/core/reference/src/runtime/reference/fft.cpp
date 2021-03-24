//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include <algorithm>
#include <cmath>
#include <queue>
#include <vector>
#include "ngraph/runtime/reference/fft.hpp"
#include "ngraph/shape.hpp"

using namespace ngraph;
using namespace ngraph::runtime::reference;

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void fft(const float* input_data,
                     const Shape& input_data_shape,
                     const int64_t* axes_data,
                     const Shape& axes_data_shape,
                     const int64_t* signal_size_data,
                     const Shape& signal_size_data_shape,
                     FFTKind fft_kind)
             {
             }

            void fft_postprocessing(const HostTensorVector& outputs,
                                    const ngraph::element::Type output_type,
                                    const std::vector<float>& fft_result)
            {
                size_t fft_result_size = fft_result.size();

                switch (output_type)
                {
                case element::Type_t::bf16:
                {
                    bfloat16* result_ptr = outputs[0]->get_data_ptr<bfloat16>();
                    for (size_t i = 0; i < fft_result_size; ++i)
                    {
                        result_ptr[i] = bfloat16(fft_result[i]);
                    }
                }
                break;
                case element::Type_t::f16:
                {
                    float16* result_ptr = outputs[0]->get_data_ptr<float16>();
                    for (size_t i = 0; i < fft_result_size; ++i)
                    {
                        result_ptr[i] = float16(fft_result[i]);
                    }
                }
                break;
                case element::Type_t::f32:
                {
                    float* result_ptr = outputs[0]->get_data_ptr<float>();
                    memcpy(result_ptr, fft_result.data(), fft_result_size * sizeof(float));
                }
                break;
                default:;
                }
            }
        }
    }
}