// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/snippets_isa.hpp"

#include <string>
#include <iostream>
#include <array>

#include "cpu_generator.hpp"
#include "jit_snippets_emitters.hpp"
#include "jit_eltwise_emitters.hpp"
#include "jit_dnnl_emitters.hpp"
#include "jit_dnnl_ext_emitters.hpp"
#include "jit_conversion_emitters.hpp"

#include "snippets_transformations/op/load_convert.hpp"
#include "snippets_transformations/op/store_convert.hpp"
#include "ngraph_transformations/op/swish_cpu.hpp"

#include <ngraph/opsets/opset5.hpp>

using namespace std;
using namespace ngraph::snippets;

#define CREATE_EMITTER(e_type) [this](const std::shared_ptr<ngraph::Node>& n) \
    -> std::shared_ptr<ngraph::snippets::Emitter> {return std::make_shared<e_type>(h.get(), isa, n);};

class jit_snippet : public dnnl::impl::cpu::x64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_snippet)

    ~jit_snippet() = default;

    jit_snippet() : jit_generator() {
    }

    void generate() override {
    }
};

ov::intel_cpu::CPUTargetMachine::CPUTargetMachine(dnnl::impl::cpu::x64::cpu_isa_t host_isa)
    : TargetMachine(), h(new jit_snippet()), isa(host_isa) {
    // data movement
    jitters[ngraph::opset1::Parameter::get_type_info_static()] = CREATE_EMITTER(NopEmitter);
    jitters[ngraph::opset1::Result::get_type_info_static()] = CREATE_EMITTER(NopEmitter);
    // jitters[ngraph::opset1::Constant::get_type_info_static()] = CREATE_EMITTER(); // Not supported

    jitters[ngraph::snippets::op::Load::get_type_info_static()] = CREATE_EMITTER(LoadEmitter);
    jitters[ngraph::snippets::op::BroadcastLoad::get_type_info_static()] = CREATE_EMITTER(BroadcastLoadEmitter);
    jitters[ov::intel_cpu::LoadConvertSaturation::get_type_info_static()] = CREATE_EMITTER(LoadConvertEmitter);
    jitters[ov::intel_cpu::LoadConvertTruncation::get_type_info_static()] = CREATE_EMITTER(LoadConvertEmitter);

    jitters[ngraph::snippets::op::Store::get_type_info_static()] = CREATE_EMITTER(StoreEmitter);
    jitters[ov::intel_cpu::StoreConvertSaturation::get_type_info_static()] = CREATE_EMITTER(StoreConvertEmitter);
    jitters[ov::intel_cpu::StoreConvertTruncation::get_type_info_static()] = CREATE_EMITTER(StoreConvertEmitter);

    jitters[ngraph::snippets::op::Scalar::get_type_info_static()] = CREATE_EMITTER(ScalarEmitter);
    jitters[ngraph::snippets::op::BroadcastMove::get_type_info_static()] = CREATE_EMITTER(BroadcastMoveEmitter);
    // jitters[ngraph::snippets::op::Nop::get_type_info_static()] = CREATE_EMITTER(NopEmitter); // Not supported
    // jitters[ngraph::opset1::Broadcast::get_type_info_static()] = CREATE_EMITTER(); // Not supported

    jitters[ngraph::snippets::op::ConvertTruncation::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_convert_truncation_emitter);
    jitters[ngraph::snippets::op::ConvertSaturation::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_convert_saturation_emitter);
    // jitters[ngraph::opset1::FakeQuantize::get_type_info_static()] = CREATE_EMITTER(); // not supported

    // binary
    jitters[ngraph::opset1::Add::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_add_emitter);
    jitters[ngraph::opset1::Divide::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_divide_emitter);
    jitters[ngraph::opset1::Equal::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_equal_emitter);
    jitters[ngraph::opset1::FloorMod::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_floor_mod_emitter);
    jitters[ngraph::opset1::Greater::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_greater_emitter);
    jitters[ngraph::opset1::GreaterEqual::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_greater_equal_emitter);
    jitters[ngraph::opset1::Less::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_less_emitter);
    jitters[ngraph::opset1::LessEqual::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_less_equal_emitter);
    jitters[ngraph::opset1::LogicalAnd::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_logical_and_emitter);
    jitters[ngraph::opset1::LogicalOr::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_logical_or_emitter);
    jitters[ngraph::opset1::LogicalXor::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_logical_xor_emitter);
    jitters[ngraph::opset1::Maximum::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_maximum_emitter);
    jitters[ngraph::opset1::Minimum::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_minimum_emitter);
    jitters[ngraph::opset1::Mod::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_mod_emitter);
    jitters[ngraph::opset1::Multiply::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_multiply_emitter);
    jitters[ngraph::opset1::NotEqual::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_not_equal_emitter);
    jitters[ngraph::snippets::op::PowerStatic::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_power_static_emitter);
    jitters[ngraph::opset1::Power::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_power_dynamic_emitter);
    jitters[ngraph::opset1::PRelu::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_prelu_emitter);
    jitters[ngraph::opset1::SquaredDifference::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_squared_difference_emitter);
    jitters[ngraph::opset1::Subtract::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_subtract_emitter);
    jitters[ngraph::opset1::Xor::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_logical_xor_emitter);

    // unary
    jitters[ngraph::opset1::Abs::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_abs_emitter);
    // jitters[ngraph::opset1::Acos::get_type_info_static()] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::Asin::get_type_info_static()] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::Atan::get_type_info_static()] = CREATE_EMITTER(); // not supported
    jitters[ngraph::opset1::Ceiling::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_ceiling_emitter);
    jitters[ngraph::opset1::Clamp::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_clamp_emitter);
    // jitters[ngraph::opset1::Cos::get_type_info_static()] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::Cosh::get_type_info_static()] = CREATE_EMITTER(); // not supported
    jitters[ngraph::opset1::Elu::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_elu_emitter);
    jitters[ngraph::opset1::Erf::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_erf_emitter);
    jitters[ngraph::opset1::Exp::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_exp_emitter);
    jitters[ngraph::opset1::Floor::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_floor_emitter);
    jitters[ngraph::opset5::Round::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_round_emitter);
    // jitters[ngraph::opset1::Log::get_type_info_static()] = CREATE_EMITTER(); // not supported
    jitters[ngraph::opset1::LogicalNot::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_logical_not_emitter);
    jitters[ngraph::opset1::Negative::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_negative_emitter);
    jitters[ngraph::opset1::Relu::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_relu_emitter);
    // jitters[ngraph::opset1::Sign::get_type_info_static()] = CREATE_EMITTER(); // not supported
    jitters[ngraph::opset1::Sigmoid::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_sigmoid_emitter);
    // jitters[ngraph::opset1::Sin::get_type_info_static()] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::Sinh::get_type_info_static()] = CREATE_EMITTER(); // not supported
    jitters[ngraph::opset1::Sqrt::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_sqrt_emitter);
    // jitters[ngraph::opset1::Tan::get_type_info_static()] = CREATE_EMITTER(); // not supported
    jitters[ngraph::opset1::Tanh::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_tanh_emitter);

    jitters[ov::intel_cpu::SwishNode::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_swish_emitter);
    jitters[ngraph::op::v4::HSwish::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_hswish_emitter);
    // jitters[ngraph::opset1::HardSigmoid::get_type_info_static()] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::Selu::get_type_info_static()] = CREATE_EMITTER(); // not supported
    jitters[ngraph::op::v0::Gelu::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_gelu_v0_emitter);
    jitters[ngraph::op::v7::Gelu::get_type_info_static()] = CREATE_EMITTER(ov::intel_cpu::jit_gelu_v7_emitter);

    jitters[ngraph::snippets::op::Kernel::get_type_info_static()] = CREATE_EMITTER(KernelEmitter);
    jitters[ngraph::snippets::op::Tile::get_type_info_static()] = CREATE_EMITTER(TileEmitter);
    jitters[ngraph::snippets::op::TileScheduler::get_type_info_static()] = CREATE_EMITTER(TileSchedulerEmitter);
}

size_t ov::intel_cpu::CPUTargetMachine::get_lanes() const {
    switch (isa) {
        case dnnl::impl::cpu::x64::avx2 : return dnnl::impl::cpu::x64::cpu_isa_traits<dnnl::impl::cpu::x64::avx2>::vlen / sizeof(float);
        case dnnl::impl::cpu::x64::sse41 : return dnnl::impl::cpu::x64::cpu_isa_traits<dnnl::impl::cpu::x64::sse41>::vlen / sizeof(float);
        case dnnl::impl::cpu::x64::avx512_core : return dnnl::impl::cpu::x64::cpu_isa_traits<dnnl::impl::cpu::x64::avx512_core>::vlen / sizeof(float);
        default : IE_THROW() << "unknown isa " << isa;
    }
}

bool ov::intel_cpu::CPUTargetMachine::is_supported() const {
    return dnnl::impl::cpu::x64::mayiuse(isa);
}

code ov::intel_cpu::CPUTargetMachine::get_snippet() const {
    h->create_kernel();
    return h->jit_ker();
}

ov::intel_cpu::CPUGenerator::CPUGenerator(dnnl::impl::cpu::x64::cpu_isa_t isa_) : Generator(std::make_shared<CPUTargetMachine>(isa_)) {
}
