// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_common.h>

#include "dnnl_descriptor.h"

namespace ov {
namespace intel_cpu {

mkldnn::primitive_desc_iterator DnnlDesriptor::createPrimitiveDescriptorIterator(const mkldnn::engine &engine,
                                                                                    const mkldnn::primitive_attr &attr) const {
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

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<mkldnn::convolution_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::convolution_forward::desc>(desc));
}

DnnlDesriptor::operator std::shared_ptr<mkldnn::convolution_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<mkldnn::convolution_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<mkldnn::deconvolution_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::deconvolution_forward::desc>(desc));
}

DnnlDesriptor::operator std::shared_ptr<mkldnn::deconvolution_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<mkldnn::deconvolution_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<mkldnn::convolution_backward_data::desc> desc,
                                   std::shared_ptr<mkldnn::convolution_forward::primitive_desc> prim) {
    this->desc.reset(
            new DescBwdImpl<mkldnn::convolution_backward_data::desc,
                    mkldnn::convolution_forward::primitive_desc>(desc, prim));
}

DnnlDesriptor::operator std::shared_ptr<mkldnn::convolution_backward_data::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescBwdImpl<mkldnn::convolution_backward_data::desc, mkldnn::convolution_forward::primitive_desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

DnnlDesriptor::operator std::shared_ptr<mkldnn::convolution_forward::primitive_desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescBwdImpl<mkldnn::convolution_backward_data::desc, mkldnn::convolution_forward::primitive_desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPrimPtr();
}

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<mkldnn::inner_product_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::inner_product_forward::desc>(desc));
}

DnnlDesriptor::operator std::shared_ptr<mkldnn::inner_product_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<mkldnn::inner_product_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<mkldnn::lrn_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::lrn_forward::desc>(desc));
}

DnnlDesriptor::operator std::shared_ptr<mkldnn::lrn_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<mkldnn::lrn_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<mkldnn::pooling_v2_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::pooling_v2_forward::desc>(desc));
}

DnnlDesriptor::operator std::shared_ptr<mkldnn::pooling_v2_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<mkldnn::pooling_v2_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<mkldnn::softmax_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::softmax_forward::desc>(desc));
}

DnnlDesriptor::operator std::shared_ptr<mkldnn::softmax_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<mkldnn::softmax_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<mkldnn::vanilla_rnn_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::vanilla_rnn_forward::desc>(desc));
}

DnnlDesriptor::operator std::shared_ptr<mkldnn::vanilla_rnn_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<mkldnn::vanilla_rnn_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<mkldnn::lstm_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::lstm_forward::desc>(desc));
}

DnnlDesriptor::operator std::shared_ptr<mkldnn::lstm_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<mkldnn::lstm_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<mkldnn::gru_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::gru_forward::desc>(desc));
}

DnnlDesriptor::operator std::shared_ptr<mkldnn::gru_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<mkldnn::gru_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<mkldnn::lbr_gru_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::lbr_gru_forward::desc>(desc));
}

DnnlDesriptor::operator std::shared_ptr<mkldnn::lbr_gru_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<mkldnn::lbr_gru_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<mkldnn::eltwise_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::eltwise_forward::desc>(desc));
}

DnnlDesriptor::operator std::shared_ptr<mkldnn::eltwise_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<mkldnn::eltwise_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

DnnlDesriptor::DnnlDesriptor(std::shared_ptr<mkldnn::matmul::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::matmul::desc>(desc));
}

DnnlDesriptor::operator std::shared_ptr<mkldnn::matmul::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<mkldnn::matmul::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

}   // namespace intel_cpu
}   // namespace ov
