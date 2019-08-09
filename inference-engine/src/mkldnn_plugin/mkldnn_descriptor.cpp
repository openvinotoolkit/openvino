// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <details/ie_exception.hpp>
#include "mkldnn_descriptor.h"

mkldnn::primitive_desc_iterator MKLDNNDescriptor::createPrimitiveDescriptorIterator(const mkldnn::engine &engine,
                                                                                    const mkldnn::primitive_attr &attr) const {
    return desc->createPrimitiveDescriptorIterator(attr, engine);
}

MKLDNNDescriptor::operator bool() {
    return desc.get() != nullptr;
}

size_t MKLDNNDescriptor::inputNumbers() const {
    DescFwdImpl<mkldnn::roi_pooling_forward::desc> *roiPooling =
            dynamic_cast<DescFwdImpl<mkldnn::roi_pooling_forward::desc> *>(desc.get());
    if (roiPooling != nullptr) {
        return roiPooling->getPtr()->c_api_inputs.size();
    }
    DescFwdImpl<mkldnn::deformable_convolution_forward::desc> *defConv =
            dynamic_cast<DescFwdImpl<mkldnn::deformable_convolution_forward::desc> *>(desc.get());
    if (defConv != nullptr) {
        return defConv->getPtr()->c_api_inputs.size();
    }
    return 1;
}

size_t MKLDNNDescriptor::outputNumbers() const {
    return 1;
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::batch_normalization_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::batch_normalization_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::batch_normalization_forward::desc>() {
    DescFwdImpl<mkldnn::batch_normalization_forward::desc> *typeDesc =
            dynamic_cast<DescFwdImpl<mkldnn::batch_normalization_forward::desc> *>(desc.get());
    if (typeDesc == nullptr) {
        THROW_IE_EXCEPTION << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::convolution_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::convolution_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::convolution_forward::desc>() {
    DescFwdImpl<mkldnn::convolution_forward::desc> *typeDesc =
            dynamic_cast<DescFwdImpl<mkldnn::convolution_forward::desc> *>(desc.get());
    if (typeDesc == nullptr) {
        THROW_IE_EXCEPTION << "Cannot cast descriptor!";
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
    DescBwdImpl<mkldnn::convolution_backward_data::desc, mkldnn::convolution_forward::primitive_desc> *typeDesc =
            dynamic_cast<DescBwdImpl<mkldnn::convolution_backward_data::desc,
                    mkldnn::convolution_forward::primitive_desc> *>(desc.get());
    if (typeDesc == nullptr) {
        THROW_IE_EXCEPTION << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::convolution_forward::primitive_desc>() {
    DescBwdImpl<mkldnn::convolution_backward_data::desc, mkldnn::convolution_forward::primitive_desc> *typeDesc =
            dynamic_cast<DescBwdImpl<mkldnn::convolution_backward_data::desc,
                    mkldnn::convolution_forward::primitive_desc> *>(desc.get());
    if (typeDesc == nullptr) {
        THROW_IE_EXCEPTION << "Cannot cast descriptor!";
    }
    return typeDesc->getPrimPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::inner_product_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::inner_product_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::inner_product_forward::desc>() {
    DescFwdImpl<mkldnn::inner_product_forward::desc> *typeDesc =
            dynamic_cast<DescFwdImpl<mkldnn::inner_product_forward::desc> *>(desc.get());
    if (typeDesc == nullptr) {
        THROW_IE_EXCEPTION << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::lrn_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::lrn_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::lrn_forward::desc>() {
    DescFwdImpl<mkldnn::lrn_forward::desc> *typeDesc =
            dynamic_cast<DescFwdImpl<mkldnn::lrn_forward::desc> *>(desc.get());
    if (typeDesc == nullptr) {
        THROW_IE_EXCEPTION << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::pooling_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::pooling_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::pooling_forward::desc>() {
    DescFwdImpl<mkldnn::pooling_forward::desc> *typeDesc =
            dynamic_cast<DescFwdImpl<mkldnn::pooling_forward::desc> *>(desc.get());
    if (typeDesc == nullptr) {
        THROW_IE_EXCEPTION << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::roi_pooling_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::roi_pooling_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::roi_pooling_forward::desc>() {
    DescFwdImpl<mkldnn::roi_pooling_forward::desc> *typeDesc =
            dynamic_cast<DescFwdImpl<mkldnn::roi_pooling_forward::desc> *>(desc.get());
    if (typeDesc == nullptr) {
        THROW_IE_EXCEPTION << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::softmax_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::softmax_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::softmax_forward::desc>() {
    DescFwdImpl<mkldnn::softmax_forward::desc> *typeDesc =
            dynamic_cast<DescFwdImpl<mkldnn::softmax_forward::desc> *>(desc.get());
    if (typeDesc == nullptr) {
        THROW_IE_EXCEPTION << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::depthwise_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::depthwise_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::depthwise_forward::desc>() {
    DescFwdImpl<mkldnn::depthwise_forward::desc> *typeDesc =
            dynamic_cast<DescFwdImpl<mkldnn::depthwise_forward::desc> *>(desc.get());
    if (typeDesc == nullptr) {
        THROW_IE_EXCEPTION << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::rnn_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::rnn_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::rnn_forward::desc>() {
    DescFwdImpl<mkldnn::rnn_forward::desc> *typeDesc =
            dynamic_cast<DescFwdImpl<mkldnn::rnn_forward::desc> *>(desc.get());
    if (typeDesc == nullptr) {
        THROW_IE_EXCEPTION << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::eltwise_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::eltwise_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::eltwise_forward::desc>() {
    DescFwdImpl<mkldnn::eltwise_forward::desc> *typeDesc =
            dynamic_cast<DescFwdImpl<mkldnn::eltwise_forward::desc> *>(desc.get());
    if (typeDesc == nullptr) {
        THROW_IE_EXCEPTION << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::binarization_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::binarization_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::binarization_forward::desc>() {
    auto *typeDesc = dynamic_cast<DescFwdImpl<mkldnn::binarization_forward::desc> *>(desc.get());
    if (typeDesc == nullptr) {
        THROW_IE_EXCEPTION << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::binary_convolution_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::binary_convolution_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::binary_convolution_forward::desc>() {
    auto *typeDesc = dynamic_cast<DescFwdImpl<mkldnn::binary_convolution_forward::desc> *>(desc.get());
    if (typeDesc == nullptr) {
        THROW_IE_EXCEPTION << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}

MKLDNNDescriptor::MKLDNNDescriptor(std::shared_ptr<mkldnn::deformable_convolution_forward::desc> desc) {
    this->desc.reset(new DescFwdImpl<mkldnn::deformable_convolution_forward::desc>(desc));
}

MKLDNNDescriptor::operator std::shared_ptr<mkldnn::deformable_convolution_forward::desc>() {
    auto *typeDesc = dynamic_cast<DescFwdImpl<mkldnn::deformable_convolution_forward::desc> *>(desc.get());
    if (typeDesc == nullptr) {
        THROW_IE_EXCEPTION << "Cannot cast descriptor!";
    }
    return typeDesc->getPtr();
}
