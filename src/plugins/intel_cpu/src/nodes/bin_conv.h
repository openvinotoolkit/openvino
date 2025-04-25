// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

struct jit_bin_conv_params {
    int mb;
    int ngroups;
    int ic, oc, ic_padded;
    int ih, iw, oh, ow;
    int l_pad, t_pad, b_pad;
    int kh, kw;
    int stride_h, stride_w;
    int dilate_h, dilate_w;
    bool with_sum;
    bool with_dw_conv;
    bool with_binarization;

    float pad_value;
    bool exclude_pad;

    int nb_ic, ic_block;
    int nb_oc, oc_block;
    int nb_oc_blocking;
    int ur_w, ur_w_tail;
    int typesize_in, typesize_out;
    dnnl::memory::data_type dst_dt;
};

struct jit_dw_conv_params {
    int kh;
};

struct jit_bin_conv_call_args {
    const void* src;
    const void* dst;
    const void* filt;
    size_t kh_padding;
    size_t kw_padding;
    size_t oc_work;
    size_t t_overflow;
    size_t b_overflow;
    size_t oc_off;
    const void** post_op_data;
};

struct jit_uni_bin_conv_kernel {
    void (*ker_)(const jit_bin_conv_call_args*);

    void operator()(const jit_bin_conv_call_args* args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_bin_conv_kernel(jit_bin_conv_params jcp,
                                     jit_dw_conv_params jcp_dw_conv,
                                     const dnnl_primitive_attr& attr)
        : ker_(nullptr),
          jcp_(jcp),
          jcp_dw_conv_(jcp_dw_conv),
          attr_(attr) {}
    virtual ~jit_uni_bin_conv_kernel() {}

    virtual void create_ker() = 0;

    jit_bin_conv_params jcp_;
    jit_dw_conv_params jcp_dw_conv_;

    const dnnl_primitive_attr& attr_;
};

class BinaryConvolution : public Node {
public:
    BinaryConvolution(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void createPrimitive() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }
    void setPostOps(dnnl::primitive_attr& attr);
    bool canFuse(const NodePtr& node) const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    impl_desc_type getImplType() {
        return implType;
    }

private:
    bool withSum = false;
    bool withBinarization = false;

    size_t group = 1;
    float pad_value = 0.f;

    std::vector<ptrdiff_t> stride;
    std::vector<ptrdiff_t> dilation;
    std::vector<ptrdiff_t> paddingL;
    std::vector<ptrdiff_t> paddingR;

    jit_bin_conv_params jcp = {};
    std::shared_ptr<jit_uni_bin_conv_kernel> bin_conv_kernel = nullptr;

    dnnl::primitive_attr attr;
    std::vector<const void*> postOpsDataPtrs;

    impl_desc_type implType = impl_desc_type::ref;

    void executeOptimized(const uint8_t* src,
                          const uint8_t* weights,
                          uint8_t* dst,
                          const std::vector<size_t>& s_str,
                          const std::vector<size_t>& w_str,
                          const std::vector<size_t>& d_str);
    void executeReference(const uint8_t* src,
                          const uint8_t* weights,
                          uint8_t* dst,
                          const std::vector<size_t>& s_str,
                          const std::vector<size_t>& w_str,
                          const std::vector<size_t>& d_str);
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
