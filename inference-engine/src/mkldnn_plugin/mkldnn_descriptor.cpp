// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_common.h>

#include "mkldnn_descriptor.h"

mkldnn::primitive_desc_iterator MKLDNNDescriptor::createPrimitiveDescriptorIterator(const mkldnn::engine &engine,
                                                                                    const mkldnn::primitive_attr &attr) const {
    return desc->createPrimitiveDescriptorIterator(attr, engine);
}

MKLDNNDescriptor::operator bool() {
    return desc != nullptr;
}

size_t MKLDNNDescriptor::inputNumbers() const {
    return 1;
}

size_t MKLDNNDescriptor::outputNumbers() const {
    return 1;
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::convolution_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::convolution_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::convolution_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<mkldnn::convolution_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::deconvolution_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::deconvolution_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::deconvolution_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<mkldnn::deconvolution_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::convolution_backward_data::desc> desc,
                                   std::shared_ptr<mkldnn::convolution_forward::primitive_desc> prim) {
    this->desc.reset(
            new DescBwdImpl<mkldnn::convolution_backward_data::desc,
                    mkldnn::convolution_forward::primitive_desc>(desc, prim));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::convolution_backward_data::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescBwdImpl<mkldnn::convolution_backward_data::desc, mkldnn::convolution_forward::primitive_desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::convolution_forward::primitive_desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescBwdImpl<mkldnn::convolution_backward_data::desc, mkldnn::convolution_forward::primitive_desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPrimPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::inner_product_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::inner_product_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::inner_product_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<mkldnn::inner_product_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::lrn_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::lrn_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::lrn_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<mkldnn::lrn_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::pooling_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::pooling_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::pooling_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<mkldnn::pooling_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::softmax_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::softmax_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::softmax_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<mkldnn::softmax_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::vanilla_rnn_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::vanilla_rnn_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::vanilla_rnn_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<mkldnn::vanilla_rnn_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::lstm_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::lstm_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::lstm_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<mkldnn::lstm_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::gru_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::gru_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::gru_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<mkldnn::gru_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::lbr_gru_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::lbr_gru_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::lbr_gru_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<mkldnn::lbr_gru_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::eltwise_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::eltwise_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::eltwise_forward::desc>() {
    auto typeDesc = std::dynamic_pointer_cast<DescFwdImpl<mkldnn::eltwise_forward::desc>>(desc);
    if (typeDesc == nullptr) {
        IE_THROW() << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}
