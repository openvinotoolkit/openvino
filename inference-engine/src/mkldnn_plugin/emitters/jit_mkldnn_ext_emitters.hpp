// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_mkldnn_emitters.hpp"

namespace MKLDNNPlugin {

class jit_relu_emitter : public jit_mkldnn_emitter {
public:
    jit_relu_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32)
        : jit_mkldnn_emitter(host, host_isa, n, exec_prc) {
            kind = mkldnn_eltwise_relu;
            alpha = 0.f;
            beta = 0.f;

            set_injector();
        }
};

class jit_sigmoid_emitter : public jit_mkldnn_emitter {
public:
    jit_sigmoid_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32)
        : jit_mkldnn_emitter(host, host_isa, n, exec_prc) {
            kind = mkldnn_eltwise_logistic;
            alpha = 0.f;
            beta = 0.f;

            set_injector();
        }
};

class jit_tanh_emitter : public jit_mkldnn_emitter {
public:
    jit_tanh_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32)
        : jit_mkldnn_emitter(host, host_isa, n, exec_prc) {
            kind = mkldnn_eltwise_tanh;
            alpha = 0.f;
            beta = 0.f;

            set_injector();
        }
};

class jit_elu_emitter : public jit_mkldnn_emitter {
public:
    jit_elu_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32)
        : jit_mkldnn_emitter(host, host_isa, n, exec_prc) {
            kind = mkldnn_eltwise_elu;
            alpha = ngraph::as_type_ptr<ngraph::opset1::Elu>(n)->get_alpha();
            beta = 0.f;

            set_injector();
        }
};

class jit_exp_emitter : public jit_mkldnn_emitter {
public:
    jit_exp_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32)
        : jit_mkldnn_emitter(host, host_isa, n, exec_prc) {
            kind = mkldnn_eltwise_exp;
            alpha = 0.f;
            beta = 0.f;

            set_injector();
        }
};

class jit_abs_emitter : public jit_mkldnn_emitter {
public:
    jit_abs_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32)
        : jit_mkldnn_emitter(host, host_isa, n, exec_prc) {
            kind = mkldnn_eltwise_abs;
            alpha = 0.f;
            beta = 0.f;

            set_injector();
        }
};

class jit_clamp_emitter : public jit_mkldnn_emitter {
public:
    jit_clamp_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const std::shared_ptr<ngraph::Node>& n,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32)
        : jit_mkldnn_emitter(host, host_isa, n, exec_prc) {
            kind = dnnl_eltwise_clip;
            auto op = ngraph::as_type_ptr<ngraph::opset1::Clamp>(n);
            alpha = op->get_min();
            beta = op->get_max();

            set_injector();
        }
};


} // namespace MKLDNNPlugin