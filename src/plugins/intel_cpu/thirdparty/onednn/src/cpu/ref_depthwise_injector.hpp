/*******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef REF_DEPTHWISE_INJECTOR_HPP
#define REF_DEPTHWISE_INJECTOR_HPP

#include "common/primitive.hpp"
#include "common/primitive_attr.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct ref_depthwise_scalar_fwd_t {
public:
    explicit ref_depthwise_scalar_fwd_t(alg_kind_t alg);
    float compute_scalar(float s, const float* weights, const float* bias) const;

private:
    alg_kind_t alg;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
