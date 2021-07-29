// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include "mkldnn/ie_mkldnn.h"

class MKLDNNDescriptor {
public:
    explicit MKLDNNDescriptor(std::shared_ptr<mkldnn::convolution_forward::desc> desc);
    operator std::shared_ptr<mkldnn::convolution_forward::desc>();

    MKLDNNDescriptor(std::shared_ptr<mkldnn::convolution_backward_data::desc> desc,
                     std::shared_ptr<mkldnn::convolution_forward::primitive_desc> prim);

    explicit MKLDNNDescriptor(std::shared_ptr<mkldnn::deconvolution_forward::desc> desc);
    operator std::shared_ptr<mkldnn::deconvolution_forward::desc>();

    operator std::shared_ptr<mkldnn::convolution_backward_data::desc>();
    operator std::shared_ptr<mkldnn::convolution_forward::primitive_desc>();

    explicit MKLDNNDescriptor(std::shared_ptr<mkldnn::inner_product_forward::desc> desc);
    operator std::shared_ptr<mkldnn::inner_product_forward::desc>();

    explicit MKLDNNDescriptor(std::shared_ptr<mkldnn::lrn_forward::desc> desc);
    operator std::shared_ptr<mkldnn::lrn_forward::desc>();

    explicit MKLDNNDescriptor(std::shared_ptr<mkldnn::pooling_forward::desc> desc);
    operator std::shared_ptr<mkldnn::pooling_forward::desc>();

    explicit MKLDNNDescriptor(std::shared_ptr<mkldnn::softmax_forward::desc> desc);
    operator std::shared_ptr<mkldnn::softmax_forward::desc>();

    explicit MKLDNNDescriptor(std::shared_ptr<mkldnn::vanilla_rnn_forward::desc> desc);
    operator std::shared_ptr<mkldnn::vanilla_rnn_forward::desc>();

    explicit MKLDNNDescriptor(std::shared_ptr<mkldnn::lstm_forward::desc> desc);
    operator std::shared_ptr<mkldnn::lstm_forward::desc>();

    explicit MKLDNNDescriptor(std::shared_ptr<mkldnn::gru_forward::desc> desc);
    operator std::shared_ptr<mkldnn::gru_forward::desc>();

    explicit MKLDNNDescriptor(std::shared_ptr<mkldnn::lbr_gru_forward::desc> desc);
    operator std::shared_ptr<mkldnn::lbr_gru_forward::desc>();

    explicit MKLDNNDescriptor(std::shared_ptr<mkldnn::eltwise_forward::desc> desc);
    operator std::shared_ptr<mkldnn::eltwise_forward::desc>();

    mkldnn::primitive_desc_iterator createPrimitiveDescriptorIterator(const mkldnn::engine &engine,
            const mkldnn::primitive_attr &attr = mkldnn::primitive_attr()) const;

    size_t outputNumbers() const;
    size_t inputNumbers() const;

    operator bool();

private:
    class IDesc {
    public:
        virtual ~IDesc() {}
        virtual mkldnn::primitive_desc_iterator createPrimitiveDescriptorIterator(const mkldnn::primitive_attr &attr,
                                                                                  const mkldnn::engine &engine) const = 0;
        static constexpr bool allow_empty = true;
    };

    template <class T>
    class DescFwdImpl: public IDesc {
        std::shared_ptr<T> desc;
    public:
        explicit DescFwdImpl(std::shared_ptr<T> d) : desc(d) {}

        mkldnn::primitive_desc_iterator createPrimitiveDescriptorIterator(const mkldnn::primitive_attr &attr,
                                                                          const mkldnn::engine &engine) const override {
            return mkldnn::primitive_desc_iterator(&desc->data, &attr, engine, nullptr, allow_empty);
        }

        std::shared_ptr<T>& getPtr() {
            return desc;
        }
    };


    template <class T, class P>
    class DescBwdImpl: public IDesc {
        std::shared_ptr<T> desc;
        std::shared_ptr<P> prim;

    public:
        DescBwdImpl(std::shared_ptr<T> d, std::shared_ptr<P> p) : desc(d), prim(p) {}

        mkldnn::primitive_desc_iterator createPrimitiveDescriptorIterator(const mkldnn::primitive_attr &attr,
                                                                          const mkldnn::engine &engine) const override {
            return mkldnn::primitive_desc_iterator(&desc->data, &attr, engine, prim.get()->get(), allow_empty);
        }

        std::shared_ptr<T>& getPtr() {
            return desc;
        }

        std::shared_ptr<P>& getPrimPtr() {
            return prim;
        }
    };

    std::shared_ptr<IDesc> desc;
};