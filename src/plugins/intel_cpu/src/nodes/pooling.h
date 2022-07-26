// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <string>
#include <memory>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

class Pooling : public Node {
public:
    Pooling(const std::shared_ptr<ov::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    void createDescriptor(const std::vector<MemoryDescPtr>& inputDesc,
                          const std::vector<MemoryDescPtr>& outputDesc) override;
    std::vector<dnnl::memory::format_tag> getAvailableFormatsForDims(const Shape &dims) const override;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void initDescriptor(const NodeConfig& config) override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

    void prepareParams() override;
    void executeDynamicImpl(dnnl::stream strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

protected:
    AttrPtr initPrimitiveAttr() override;

private:
    void setPostOps(dnnl::primitive_attr &attr);

    void initEffectiveAttributes(const Shape &inDims, const Shape &outDims);
    dnnl::algorithm getPoolingAlgorithm() const;
    std::shared_ptr<dnnl::pooling_v2_forward::desc> createDescriptorInternal(const dnnl::memory::desc& in_candidate,
                                                                               const dnnl::memory::desc& out_candidate,
                                                                               const dnnl::algorithm alg) const;

    AttrPtr pAttr;

    Shape inShape;

    bool isMaxPool8 = false;
    bool auto_pad = false;
    bool exclude_pad = false;
    std::vector<ptrdiff_t> dilation;
    std::vector<ptrdiff_t> stride;
    std::vector<ptrdiff_t> kernel;

    /// Effective padding. Used to define correct output shape by oneDNN
    /// reshape formula: (iw - kernel + pad_l + pad_r) / strides[i - 2] + 1
    /// should be passed into pooling desc constructor.
    std::vector<ptrdiff_t> effective_pad_begin;
    std::vector<ptrdiff_t> effective_pad_end;

    /// Effective dilation. Used to define correct dilation for OneDNN.
    /// For OneDNN default dilation is vector of zero
    std::vector<ptrdiff_t> effective_dilation;

    /// Effective pad value. Describe how much zero element added to input
    /// data tensor. May be less than "Effective padding" values.
    /// If pooling window is out of this padding, the region of averaging
    /// is decreased.
    std::vector<ptrdiff_t> data_pad_begin;
    std::vector<ptrdiff_t> data_pad_end;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
