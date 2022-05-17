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

#include "cpu/aarch64/acl_inner_product_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace acl_inner_product_utils {

using namespace format_tag;
using namespace utils;
using namespace status;

status_t init_conf_ip(acl_ip_conf_t &aip, memory_desc_t &src_md,
        memory_desc_t &wei_md, memory_desc_t &dst_md, memory_desc_t &bias_md,
        const inner_product_desc_t &ipd, const primitive_attr_t &attr) {
    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper wei_d(&wei_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bia_d(&bias_md);

    // Compute Library currently supports forward propagation only
    const prop_kind_t prop_kind = ipd.prop_kind;
    const bool is_fwd = (prop_kind == dnnl_forward_training)
            || (prop_kind == dnnl_forward_inference);
    if (!is_fwd) return status::unimplemented;

    const int with_groups = wei_d.ndims() == src_d.ndims() + 1;
    const int ndims = src_d.ndims();

    // There are two sub-cases: src & wei tensors are either 2- or 4-dimensional
    const bool is_2d = (ndims == 2) && (wei_d.ndims() == 2);
    const bool is_4d = (ndims == 4) && (wei_d.ndims() == 4);

    // Compute Library unsupported shape scenarios
    // FP32 only is supported at the moment
    if (one_of(true, !(is_4d || is_2d), with_groups)) { return unimplemented; }

    // batch size
    const int mb = src_d.dims()[0];

    // src/input channels, height, width
    const int ic = src_d.dims()[1];
    const int ih = is_4d ? src_d.dims()[ndims - 2] : 0;
    const int iw = is_4d ? src_d.dims()[ndims - 1] : 0;

    // dst/output channels
    const int oc = dst_d.dims()[1];

    // weights height, width
    const int kh = is_4d ? wei_d.dims()[with_groups + ndims - 2] : 0;
    const int kw = is_4d ? wei_d.dims()[with_groups + ndims - 1] : 0;

    aip.with_bias = ipd.bias_desc.format_kind != format_kind::undef;

    // Data layout is already defined thus should only be checked
    auto src_tag = memory_desc_matches_one_of_tag(src_md, nhwc, nchw, nc, cn);
    auto wei_tag = memory_desc_matches_one_of_tag(wei_md, ohwi, oihw, oi, io);
    auto dst_tag = memory_desc_matches_one_of_tag(dst_md, nc, cn);
    if (one_of(format_tag::undef, src_tag, wei_tag, dst_tag)) {
        return status::unimplemented;
    }

    arm_compute::TensorShape src_shape {(src_tag == nc)
                    ? arm_compute::TensorShape(ic, mb)
                    : arm_compute::TensorShape(mb, ic)};
    if (is_4d) {
        src_shape = (src_tag == nhwc)
                ? arm_compute::TensorShape(ic, iw, ih, mb)
                : arm_compute::TensorShape(iw, ih, ic, mb);
    }

    // Compute Library requires the weights to be 2-dimensional for FC layer
    arm_compute::TensorShape wei_shape {
            arm_compute::TensorShape(is_4d ? ic * kh * kw : ic, oc)};
    if (is_2d && wei_tag == io) {
        wei_shape = arm_compute::TensorShape(oc, ic);
    }

    arm_compute::DataLayout wei_layout {(wei_tag == ohwi || wei_tag == oi)
                    ? arm_compute::DataLayout::NHWC
                    : arm_compute::DataLayout::NCHW};

    // clang-format off
    aip.src_info = arm_compute::TensorInfo(
            src_shape,
            1,
            arm_compute::DataType::F32,
            (src_tag == nhwc || src_tag == nc) ?
            arm_compute::DataLayout::NHWC : arm_compute::DataLayout::NCHW);

    aip.wei_info = arm_compute::TensorInfo(
            wei_shape,
            1,
            arm_compute::DataType::F32,
            wei_layout);

    aip.dst_info = arm_compute::TensorInfo(
            (dst_tag == nhwc || dst_tag == nc) ?
            arm_compute::TensorShape(oc, mb) : arm_compute::TensorShape(mb, oc),
            1,
            arm_compute::DataType::F32,
            (dst_tag == nhwc || dst_tag == nc) ?
            arm_compute::DataLayout::NHWC : arm_compute::DataLayout::NCHW);

    aip.bia_info = arm_compute::TensorInfo(
            aip.with_bias ?
            arm_compute::TensorShape(oc) : arm_compute::TensorShape(),
            1,
            arm_compute::DataType::F32);
    // clang-format on

    aip.fc_info.weights_trained_layout = wei_layout;
    if (is_2d && wei_tag != src_tag) { aip.fc_info.transpose_weights = false; }

    // Either activation or sum is supported as post-op at the moment
    aip.fc_info.activation_info = acl_common_utils::get_acl_act(attr);
    const auto &post_ops = attr.post_ops_;
    aip.with_sum = (post_ops.len() == 1) && post_ops.entry_[0].is_sum();

    // clang-format off
    // Validate convolution manually to check for return status
    auto acl_st = arm_compute::NEFullyConnectedLayer::validate(
        &aip.src_info,
        &aip.wei_info,
        aip.with_bias ? &aip.bia_info : nullptr,
        &aip.dst_info,
	aip.fc_info);
    // clang-format on
    if (acl_st.error_code() != arm_compute::ErrorCode::OK) {
        return status::unimplemented;
    }

    return status::success;
}

} // namespace acl_inner_product_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
