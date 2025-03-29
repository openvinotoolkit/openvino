// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "jit_dnnl_emitters.hpp"
#include "openvino/opsets/opset5.hpp"
#include "transformations/cpu_opset/common/op/swish_cpu.hpp"
#include "utils/ngraph_utils.hpp"

namespace ov::intel_cpu {

class jit_relu_emitter : public jit_dnnl_emitter {
public:
    jit_relu_emitter(dnnl::impl::cpu::x64::jit_generator* host,
                     dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                     const std::shared_ptr<ov::Node>& n,
                     ov::element::Type exec_prc = ov::element::f32)
        : jit_dnnl_emitter(host, host_isa, n, exec_prc) {
        kind = dnnl_eltwise_relu;
        alpha = 0.f;
        beta = 0.f;

        set_injector();
    }
};

class jit_sigmoid_emitter : public jit_dnnl_emitter {
public:
    jit_sigmoid_emitter(dnnl::impl::cpu::x64::jit_generator* host,
                        dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                        const std::shared_ptr<ov::Node>& n,
                        ov::element::Type exec_prc = ov::element::f32)
        : jit_dnnl_emitter(host, host_isa, n, exec_prc) {
        kind = dnnl_eltwise_logistic;
        alpha = 0.f;
        beta = 0.f;

        set_injector();
    }
};

class jit_tanh_emitter : public jit_dnnl_emitter {
public:
    jit_tanh_emitter(dnnl::impl::cpu::x64::jit_generator* host,
                     dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                     const std::shared_ptr<ov::Node>& n,
                     ov::element::Type exec_prc = ov::element::f32)
        : jit_dnnl_emitter(host, host_isa, n, exec_prc) {
        kind = dnnl_eltwise_tanh;
        alpha = 0.f;
        beta = 0.f;

        set_injector();
    }
};

class jit_elu_emitter : public jit_dnnl_emitter {
public:
    jit_elu_emitter(dnnl::impl::cpu::x64::jit_generator* host,
                    dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                    const std::shared_ptr<ov::Node>& n,
                    ov::element::Type exec_prc = ov::element::f32)
        : jit_dnnl_emitter(host, host_isa, n, exec_prc) {
        kind = dnnl_eltwise_elu;
        alpha = ov::as_type_ptr<ov::op::v0::Elu>(n)->get_alpha();
        beta = 0.f;

        set_injector();
    }
};

class jit_abs_emitter : public jit_dnnl_emitter {
public:
    jit_abs_emitter(dnnl::impl::cpu::x64::jit_generator* host,
                    dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                    const std::shared_ptr<ov::Node>& n,
                    ov::element::Type exec_prc = ov::element::f32)
        : jit_dnnl_emitter(host, host_isa, n, exec_prc) {
        kind = dnnl_eltwise_abs;
        alpha = 0.f;
        beta = 0.f;

        set_injector();
    }
};

class jit_clamp_emitter : public jit_dnnl_emitter {
public:
    jit_clamp_emitter(dnnl::impl::cpu::x64::jit_generator* host,
                      dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                      const std::shared_ptr<ov::Node>& n,
                      ov::element::Type exec_prc = ov::element::f32)
        : jit_dnnl_emitter(host, host_isa, n, exec_prc) {
        kind = dnnl_eltwise_clip;
        auto op = ov::as_type_ptr<ov::op::v0::Clamp>(n);
        alpha = op->get_min();
        beta = op->get_max();

        set_injector();
    }
};

class jit_swish_emitter : public jit_dnnl_emitter {
public:
    jit_swish_emitter(dnnl::impl::cpu::x64::jit_generator* host,
                      dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                      const std::shared_ptr<ov::Node>& n,
                      ov::element::Type exec_prc = ov::element::f32)
        : jit_dnnl_emitter(host, host_isa, n, exec_prc) {
        kind = dnnl_eltwise_swish;
        auto op = ov::as_type_ptr<ov::intel_cpu::SwishNode>(n);
        alpha = op->get_alpha();
        beta = 0.f;

        set_injector();
    }
};

class jit_hswish_emitter : public jit_dnnl_emitter {
public:
    jit_hswish_emitter(dnnl::impl::cpu::x64::jit_generator* host,
                       dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                       const std::shared_ptr<ov::Node>& n,
                       ov::element::Type exec_prc = ov::element::f32)
        : jit_dnnl_emitter(host, host_isa, n, exec_prc) {
        // since v3.0 oneDNN has flexible version of hardswish, ov still uses the one with hardcoded alpha and beta
        kind = dnnl_eltwise_hardswish;
        alpha = 1.f / 6.f;
        beta = 0.5f;

        set_injector();
    }
};

class jit_gelu_v0_emitter : public jit_dnnl_emitter {
public:
    jit_gelu_v0_emitter(dnnl::impl::cpu::x64::jit_generator* host,
                        dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                        const std::shared_ptr<ov::Node>& n,
                        ov::element::Type exec_prc = ov::element::f32)
        : jit_dnnl_emitter(host, host_isa, n, exec_prc) {
        kind = dnnl_eltwise_gelu_erf;

        set_injector();
    }
};

class jit_gelu_v7_emitter : public jit_dnnl_emitter {
public:
    jit_gelu_v7_emitter(dnnl::impl::cpu::x64::jit_generator* host,
                        dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                        const std::shared_ptr<ov::Node>& n,
                        ov::element::Type exec_prc = ov::element::f32)
        : jit_dnnl_emitter(host, host_isa, n, exec_prc) {
        auto gelu = getNgraphOpAs<ov::op::v7::Gelu>(n);
        ov::op::GeluApproximationMode approximationMode = gelu->get_approximation_mode();
        if (approximationMode == ov::op::GeluApproximationMode::ERF) {
            kind = dnnl_eltwise_gelu_erf;
        } else if (approximationMode == ov::op::GeluApproximationMode::TANH) {
            kind = dnnl_eltwise_gelu_tanh;
        } else {
            OPENVINO_THROW_NOT_IMPLEMENTED(
                "Subgraph node doesn't support ngraph operation Gelu with approximation mode: ",
                approximationMode);
        }

        set_injector();
    }
};

class jit_round_emitter : public jit_dnnl_emitter {
public:
    jit_round_emitter(dnnl::impl::cpu::x64::jit_generator* host,
                      dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                      const std::shared_ptr<ov::Node>& n,
                      ov::element::Type exec_prc = ov::element::f32)
        : jit_dnnl_emitter(host, host_isa, n, exec_prc) {
        const auto round = getNgraphOpAs<ov::op::v5::Round>(n);
        const auto mode = round->get_mode();
        if ((mode != ov::opset5::Round::RoundMode::HALF_AWAY_FROM_ZERO) &&
            (mode != ov::opset5::Round::RoundMode::HALF_TO_EVEN)) {
            OPENVINO_THROW_NOT_IMPLEMENTED("Round emitter doesn't support ngraph operation Round with mode: ",
                                           static_cast<int>(mode));
        }

        kind = mode == ov::opset5::Round::RoundMode::HALF_AWAY_FROM_ZERO ? dnnl_eltwise_round_half_away_from_zero
                                                                         : dnnl_eltwise_round_half_to_even;
        set_injector();
    }
};

}  // namespace ov::intel_cpu
