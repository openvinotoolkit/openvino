/*******************************************************************************
* Copyright 2021 Arm Ltd. and affiliates
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

#include "cpu/aarch64/acl_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace acl_common_utils {

using namespace dnnl::impl::alg_kind;
using namespace data_type;

arm_compute::DataType get_acl_data_t(const dnnl_data_type_t dt) {
    switch (dt) {
        case bf16: return arm_compute::DataType::BFLOAT16; break;
        case f32: return arm_compute::DataType::F32; break;
        case s32: return arm_compute::DataType::S32; break;
        case f16: return arm_compute::DataType::F16; break;
        case s8: return arm_compute::DataType::QASYMM8_SIGNED; break;
        case u8: return arm_compute::DataType::QASYMM8; break;
        default: return arm_compute::DataType::UNKNOWN;
    }
}

arm_compute::ActivationLayerInfo convert_to_acl_act(
        const alg_kind_t eltwise_alg, const float alpha, const float beta) {
    using acl_act_t = arm_compute::ActivationLayerInfo::ActivationFunction;
    acl_act_t acl_act_alg;

    switch (eltwise_alg) {
        case eltwise_relu:
            // oneDNN defines RELU: f(x) = (x > 0) ? x : a*x
            // Compute Library defines LEAKY_RELU: f(x) = (x > 0) ? x : a*x
            // whilst Compute Library RELU is defined as: f(x) = max(0,x)
            if (alpha == 0) {
                acl_act_alg = acl_act_t::RELU;
            } else {
                acl_act_alg = acl_act_t::LEAKY_RELU;
            }
            break;
        case eltwise_tanh:
            // oneDNN defines TANH activation as:          f(x) = tanh(x)
            // Compute Library defines TANH activation as: f(x) = a*tanh(b*x)
            // Setting a=b=1 makes the two equivalent
            return arm_compute::ActivationLayerInfo(acl_act_t::TANH, 1.f, 1.f);
            break;
        case eltwise_elu: acl_act_alg = acl_act_t::ELU; break;
        case eltwise_square: acl_act_alg = acl_act_t::SQUARE; break;
        case eltwise_abs: acl_act_alg = acl_act_t::ABS; break;
        case eltwise_sqrt: acl_act_alg = acl_act_t::SQRT; break;
        case eltwise_linear: acl_act_alg = acl_act_t::LINEAR; break;
        case eltwise_bounded_relu: acl_act_alg = acl_act_t::BOUNDED_RELU; break;
        case eltwise_soft_relu: acl_act_alg = acl_act_t::SOFT_RELU; break;
        case eltwise_logistic: acl_act_alg = acl_act_t::LOGISTIC; break;
        default: return arm_compute::ActivationLayerInfo();
    }

    return arm_compute::ActivationLayerInfo(acl_act_alg, alpha, beta);
}

arm_compute::ActivationLayerInfo get_acl_act(const primitive_attr_t &attr) {
    const auto &post_ops = attr.post_ops_;
    const int entry_idx = post_ops.find(primitive_kind::eltwise);
    if (entry_idx == -1) { return arm_compute::ActivationLayerInfo(); }

    const auto eltwise_alg = post_ops.entry_[entry_idx].eltwise.alg;
    float alpha = post_ops.entry_[entry_idx].eltwise.alpha;
    float beta = post_ops.entry_[entry_idx].eltwise.beta;

    return convert_to_acl_act(eltwise_alg, alpha, beta);
}

arm_compute::ActivationLayerInfo get_acl_act(const eltwise_desc_t &ed) {
    const alg_kind_t eltwise_alg = ed.alg_kind;
    float alpha = ed.alpha;
    float beta = ed.beta;

    return convert_to_acl_act(eltwise_alg, alpha, beta);
}

bool acl_act_ok(alg_kind_t eltwise_activation) {
    return utils::one_of(eltwise_activation, eltwise_relu, eltwise_tanh,
            eltwise_elu, eltwise_square, eltwise_abs, eltwise_sqrt,
            eltwise_linear, eltwise_bounded_relu, eltwise_soft_relu,
            eltwise_logistic);
}

void acl_thread_bind() {
    static std::once_flag flag_once;
    // The threads in Compute Library are bound for the cores 0..max_threads-1
    // dnnl_get_max_threads() returns OMP_NUM_THREADS
    const int max_threads = dnnl_get_max_threads();
    // arm_compute::Scheduler does not support concurrent access thus a
    // workaround here restricts it to only one call
    std::call_once(flag_once, [&]() {
        arm_compute::Scheduler::get().set_num_threads(max_threads);
    });
}

} // namespace acl_common_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
