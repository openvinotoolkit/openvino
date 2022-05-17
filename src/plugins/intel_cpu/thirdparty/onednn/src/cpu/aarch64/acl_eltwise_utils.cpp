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

#include "cpu/aarch64/acl_eltwise_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::alg_kind;
using namespace prop_kind;
using namespace data_type;
using uint = unsigned int;

namespace acl_eltwise_utils {

status_t acl_eltwise_check(acl_eltwise_conf_t &aep, memory_desc_t &data_md,
        const eltwise_desc_t &ed, const primitive_attr_t &attr) {

    const memory_desc_wrapper data_d(&data_md);

    const int ndims = data_d.ndims();
    const bool is_1d = ndims == 3;
    const bool is_3d = ndims == 5;
    const bool is_int8 = one_of(ed.data_desc.data_type, s8, u8);
    bool is_nspc {true};

    // Compute Library unsupported shape scenarios
    if (one_of(true, is_3d, is_1d)) { return status::unimplemented; }

    const alg_kind_t eltwise_alg = ed.alg_kind;

    bool activation_supported = acl_common_utils::acl_act_ok(eltwise_alg);
    if (!activation_supported) { return status::unimplemented; }

    // batch size
    const int mb = data_d.dims()[0];

    // src/dst channels, height, width
    const int ic = data_d.dims()[1];
    const int ih = data_d.dims()[ndims - 2];
    const int iw = data_d.dims()[ndims - 1];

    const int oc = ic;
    const int oh = ih;
    const int ow = iw;

    auto data_tag = memory_desc_matches_one_of_tag(
            data_md, format_tag::nhwc, format_tag::nchw);
    if (data_tag == format_tag::undef) { return status::unimplemented; }

    is_nspc = utils::one_of(data_tag, format_tag::nhwc);
    const auto acl_layout = is_nspc ? arm_compute::DataLayout::NHWC
                                    : arm_compute::DataLayout::NCHW;

    auto acl_src_data_t = acl_common_utils::get_acl_data_t(data_d.data_type());
    auto acl_dst_data_t = acl_common_utils::get_acl_data_t(data_d.data_type());

    // clang-format off
    aep.src_info = arm_compute::TensorInfo(
            is_nspc ? arm_compute::TensorShape(ic, iw, ih, mb) :
            arm_compute::TensorShape(iw, ih, ic, mb),
            1,
            acl_src_data_t,
            acl_layout);

    aep.dst_info = arm_compute::TensorInfo(
            is_nspc ? arm_compute::TensorShape(oc, ow, oh, mb) :
            arm_compute::TensorShape(ow, oh, oc, mb),
            1,
            acl_dst_data_t,
            acl_layout);
    // clang-format on

    if (is_int8) {
        aep.src_info.set_quantization_info(arm_compute::QuantizationInfo(1, 0));
        aep.dst_info.set_quantization_info(arm_compute::QuantizationInfo(1, 0));
    }

    aep.act_info = acl_common_utils::get_acl_act(ed);

    return status::success;
}

status_t init_conf_eltwise(acl_eltwise_conf_t &aep, memory_desc_t &data_md,
        const eltwise_desc_t &ed, const primitive_attr_t &attr) {

    // General Compute Library checks
    CHECK(acl_eltwise_check(aep, data_md, ed, attr));

    // clang-format off
    auto acl_st = arm_compute::NEActivationLayer::validate(
        &aep.src_info,
        &aep.dst_info,
        aep.act_info);
    // clang-format on
    if (acl_st.error_code() != arm_compute::ErrorCode::OK) {
        return status::unimplemented;
    }

    return status::success;
}

} // namespace acl_eltwise_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
