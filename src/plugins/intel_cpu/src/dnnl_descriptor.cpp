// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_common.h>

#include "dnnl_descriptor.h"

namespace ov {
namespace intel_cpu {

dnnl::primitive_desc_iterator DnnlDesriptor::createPrimitiveDescriptorIterator(const dnnl::engine &engine,
                                                                                    const dnnl::primitive_attr &attr) const {
    return desc->createPrimitiveDescriptorIterator(attr, engine);
}

DnnlDesriptor::operator bool() {
    return desc != nullptr;
}

size_t DnnlDesriptor::inputNumbers() const {
    return 1;
}

size_t DnnlDesriptor::outputNumbers() const {
    return 1;
}

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<dnnl::convolution_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<dnnl::convolution_forward::desc>(desc));
}

DnnlDesriptor::operator std::shared_ptr<dnnl::convolution_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<dnnl::convolution_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<dnnl::deconvolution_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<dnnl::deconvolution_forward::desc>(desc));
}

DnnlDesriptor::operator std::shared_ptr<dnnl::deconvolution_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<dnnl::deconvolution_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<dnnl::convolution_backward_data::desc> desc,
                                   std::shared_ptr<dnnl::convolution_forward::primitive_desc> prim) {
    this->desc.reset(
            new DescBwdImpl<dnnl::convolution_backward_data::desc,
                    dnnl::convolution_forward::primitive_desc>(desc, prim));
}

DnnlDesriptor::operator std::shared_ptr<dnnl::convolution_backward_data::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescBwdImpl<dnnl::convolution_backward_data::desc, dnnl::convolution_forward::primitive_desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

DnnlDesriptor::operator std::shared_ptr<dnnl::convolution_forward::primitive_desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescBwdImpl<dnnl::convolution_backward_data::desc, dnnl::convolution_forward::primitive_desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPrimPtr();
}

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<dnnl::inner_product_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<dnnl::inner_product_forward::desc>(desc));
}

DnnlDesriptor::operator std::shared_ptr<dnnl::inner_product_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<dnnl::inner_product_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<dnnl::lrn_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<dnnl::lrn_forward::desc>(desc));
}

DnnlDesriptor::operator std::shared_ptr<dnnl::lrn_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<dnnl::lrn_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<dnnl::pooling_v2_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<dnnl::pooling_v2_forward::desc>(desc));
}

DnnlDesriptor::operator std::shared_ptr<dnnl::pooling_v2_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<dnnl::pooling_v2_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<dnnl::softmax_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<dnnl::softmax_forward::desc>(desc));
}

DnnlDesriptor::operator std::shared_ptr<dnnl::softmax_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<dnnl::softmax_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<dnnl::vanilla_rnn_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<dnnl::vanilla_rnn_forward::desc>(desc));
}

DnnlDesriptor::operator std::shared_ptr<dnnl::vanilla_rnn_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<dnnl::vanilla_rnn_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<dnnl::lstm_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<dnnl::lstm_forward::desc>(desc));
}

DnnlDesriptor::operator std::shared_ptr<dnnl::lstm_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<dnnl::lstm_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<dnnl::gru_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<dnnl::gru_forward::desc>(desc));
}

DnnlDesriptor::operator std::shared_ptr<dnnl::gru_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<dnnl::gru_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<dnnl::lbr_gru_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<dnnl::lbr_gru_forward::desc>(desc));
}

DnnlDesriptor::operator std::shared_ptr<dnnl::lbr_gru_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<dnnl::lbr_gru_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<dnnl::augru_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<dnnl::augru_forward::desc>(desc));
}

DnnlDesriptor::operator std::shared_ptr<dnnl::augru_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<dnnl::augru_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<dnnl::lbr_augru_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<dnnl::lbr_augru_forward::desc>(desc));
}

DnnlDesriptor::operator std::shared_ptr<dnnl::lbr_augru_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<dnnl::lbr_augru_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<dnnl::eltwise_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<dnnl::eltwise_forward::desc>(desc));
}

DnnlDesriptor::operator std::shared_ptr<dnnl::eltwise_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<dnnl::eltwise_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<dnnl::matmul::desc> desc) {
    this->desc.reset(new DescFwdImpl<dnnl::matmul::desc>(desc));
}

DnnlDesriptor::operator std::shared_ptr<dnnl::matmul::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<dnnl::matmul::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

}   // namespace intel_cpu
}   // namespace ov
