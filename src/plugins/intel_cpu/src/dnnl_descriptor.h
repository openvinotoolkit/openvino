// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include "onednn/dnnl.h"

namespace ov {
namespace intel_cpu {

class DnnlDesriptor {
public:
    explicit DnnlDesriptor(std::shared_ptr<dnnl::convolution_forward::desc> desc);
    operator std::shared_ptr<dnnl::convolution_forward::desc>();

    DnnlDesriptor(std::shared_ptr<dnnl::convolution_backward_data::desc> desc,
               std::shared_ptr<dnnl::convolution_forward::primitive_desc> prim);

    explicit DnnlDesriptor(std::shared_ptr<dnnl::deconvolution_forward::desc> desc);
    operator std::shared_ptr<dnnl::deconvolution_forward::desc>();

    operator std::shared_ptr<dnnl::convolution_backward_data::desc>();
    operator std::shared_ptr<dnnl::convolution_forward::primitive_desc>();

    explicit DnnlDesriptor(std::shared_ptr<dnnl::inner_product_forward::desc> desc);
    operator std::shared_ptr<dnnl::inner_product_forward::desc>();

    explicit DnnlDesriptor(std::shared_ptr<dnnl::lrn_forward::desc> desc);
    operator std::shared_ptr<dnnl::lrn_forward::desc>();

    explicit DnnlDesriptor(std::shared_ptr<dnnl::pooling_v2_forward::desc> desc);
    operator std::shared_ptr<dnnl::pooling_v2_forward::desc>();

    explicit DnnlDesriptor(std::shared_ptr<dnnl::softmax_forward::desc> desc);
    operator std::shared_ptr<dnnl::softmax_forward::desc>();

    explicit DnnlDesriptor(std::shared_ptr<dnnl::vanilla_rnn_forward::desc> desc);
    operator std::shared_ptr<dnnl::vanilla_rnn_forward::desc>();

    explicit DnnlDesriptor(std::shared_ptr<dnnl::lstm_forward::desc> desc);
    operator std::shared_ptr<dnnl::lstm_forward::desc>();

    explicit DnnlDesriptor(std::shared_ptr<dnnl::gru_forward::desc> desc);
    operator std::shared_ptr<dnnl::gru_forward::desc>();

    explicit DnnlDesriptor(std::shared_ptr<dnnl::lbr_gru_forward::desc> desc);
    operator std::shared_ptr<dnnl::lbr_gru_forward::desc>();

    explicit DnnlDesriptor(std::shared_ptr<dnnl::augru_forward::desc> desc);
    operator std::shared_ptr<dnnl::augru_forward::desc>();

    explicit DnnlDesriptor(std::shared_ptr<dnnl::lbr_augru_forward::desc> desc);
    operator std::shared_ptr<dnnl::lbr_augru_forward::desc>();

    explicit DnnlDesriptor(std::shared_ptr<dnnl::eltwise_forward::desc> desc);
    operator std::shared_ptr<dnnl::eltwise_forward::desc>();

    explicit DnnlDesriptor(std::shared_ptr<dnnl::matmul::desc> desc);
    operator std::shared_ptr<dnnl::matmul::desc>();

    dnnl::primitive_desc_iterator createPrimitiveDescriptorIterator(const dnnl::engine &engine,
            const dnnl::primitive_attr &attr = dnnl::primitive_attr()) const;

    size_t outputNumbers() const;
    size_t inputNumbers() const;

    operator bool();

private:
    class IDesc {
    public:
        virtual ~IDesc() {}
        virtual dnnl::primitive_desc_iterator createPrimitiveDescriptorIterator(const dnnl::primitive_attr &attr,
                                                                                  const dnnl::engine &engine) const = 0;
        static constexpr bool allow_empty = true;
    };

    template <class T>
    class DescFwdImpl: public IDesc {
        std::shared_ptr<T> desc;
    public:
        explicit DescFwdImpl(std::shared_ptr<T> d) : desc(d) {}

        dnnl::primitive_desc_iterator createPrimitiveDescriptorIterator(const dnnl::primitive_attr &attr,
                                                                        const dnnl::engine &engine) const override {
            return dnnl::primitive_desc_iterator(&desc->data, &attr, engine, nullptr, allow_empty);
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

        dnnl::primitive_desc_iterator createPrimitiveDescriptorIterator(const dnnl::primitive_attr &attr,
                                                                          const dnnl::engine &engine) const override {
            return dnnl::primitive_desc_iterator(&desc->data, &attr, engine, prim.get()->get(), allow_empty);
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

}   // namespace intel_cpu
}   // namespace ov
