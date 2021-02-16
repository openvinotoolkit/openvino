// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/snippets_isa.hpp"

#include <string>
#include <iostream>
#include <array>

#include "cpu_generator.hpp"
#include "jit_snippets_emitters.hpp"
#include "jit_eltwise_emitters.hpp"
#include "jit_mkldnn_emitters.hpp"
#include "jit_mkldnn_ext_emitters.hpp"

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

MKLDNNPlugin::CPUTarget::CPUTarget(dnnl::impl::cpu::x64::cpu_isa_t host_isa)
    : TargetMachine(), h(new jit_snippet()), isa(host_isa) {
    // data movement
    jitters[ngraph::opset1::Parameter::type_info] = CREATE_EMITTER(NopEmitter);
    jitters[ngraph::snippets::op::BlockedParameter::type_info] = CREATE_EMITTER(NopEmitter);
    jitters[ngraph::opset1::Result::type_info] = CREATE_EMITTER(NopEmitter);
    // jitters[ngraph::opset1::Constant::type_info] = CREATE_EMITTER(); // Not supported

    jitters[ngraph::snippets::op::Load::type_info] = CREATE_EMITTER(LoadEmitter);
    jitters[ngraph::snippets::op::VectorLoad::type_info] = CREATE_EMITTER(LoadEmitter);
    jitters[ngraph::snippets::op::ScalarLoad::type_info] = CREATE_EMITTER(ScalarLoadEmitter);
    jitters[ngraph::snippets::op::BroadcastLoad::type_info] = CREATE_EMITTER(BroadcastLoadEmitter);

    jitters[ngraph::snippets::op::Store::type_info] = CREATE_EMITTER(StoreEmitter);
    jitters[ngraph::snippets::op::VectorStore::type_info] = CREATE_EMITTER(StoreEmitter);
    jitters[ngraph::snippets::op::ScalarStore::type_info] = CREATE_EMITTER(ScalarStoreEmitter);

    jitters[ngraph::snippets::op::Scalar::type_info] = CREATE_EMITTER(ScalarEmitter);
    jitters[ngraph::snippets::op::BroadcastMove::type_info] = CREATE_EMITTER(FakeBroadcastEmitter);
    // jitters[ngraph::snippets::op::Nop::type_info] = CREATE_EMITTER(NopEmitter); // Not supported
    // jitters[ngraph::opset1::Broadcast::type_info] = CREATE_EMITTER(); // Not supported

    // jitters[ngraph::opset1::Convert::type_info] = CREATE_EMITTER(); // Not supported
    // jitters[ngraph::opset1::FakeQuantize::type_info] = CREATE_EMITTER(); // not supported

    // binary
    jitters[ngraph::opset1::Add::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_add_emitter);
    jitters[ngraph::opset1::Divide::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_divide_emitter);
    jitters[ngraph::opset1::Equal::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_equal_emitter);
    jitters[ngraph::opset1::FloorMod::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_floor_mod_emitter);
    jitters[ngraph::opset1::Greater::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_greater_emitter);
    jitters[ngraph::opset1::GreaterEqual::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_greater_equal_emitter);
    jitters[ngraph::opset1::Less::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_less_emitter);
    jitters[ngraph::opset1::LessEqual::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_less_equal_emitter);
    jitters[ngraph::opset1::LogicalAnd::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_logical_and_emitter);
    jitters[ngraph::opset1::LogicalOr::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_logical_or_emitter);
    jitters[ngraph::opset1::LogicalXor::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_logical_xor_emitter);
    jitters[ngraph::opset1::Maximum::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_maximum_emitter);
    jitters[ngraph::opset1::Minimum::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_minimum_emitter);
    jitters[ngraph::opset1::Mod::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_mod_emitter);
    jitters[ngraph::opset1::Multiply::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_multiply_emitter);
    jitters[ngraph::opset1::NotEqual::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_not_equal_emitter);
    jitters[ngraph::snippets::op::PowerStatic::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_power_static_emitter);
    jitters[ngraph::opset1::Power::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_power_dynamic_emitter);
    jitters[ngraph::opset1::PRelu::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_prelu_emitter);
    jitters[ngraph::opset1::SquaredDifference::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_squared_difference_emitter);
    jitters[ngraph::opset1::Subtract::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_subtract_emitter);
    jitters[ngraph::opset1::Xor::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_logical_xor_emitter);

    // unary
    jitters[ngraph::opset1::Abs::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_abs_emitter);
    // jitters[ngraph::opset1::Acos::type_info] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::Asin::type_info] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::Atan::type_info] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::Ceiling::type_info] = CREATE_EMITTER(); // not supported
    jitters[ngraph::opset1::Clamp::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_clamp_emitter);
    // jitters[ngraph::opset1::Cos::type_info] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::Cosh::type_info] = CREATE_EMITTER(); // not supported
    jitters[ngraph::opset1::Elu::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_elu_emitter);
    jitters[ngraph::opset1::Erf::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_erf_emitter);
    jitters[ngraph::opset1::Exp::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_exp_emitter);
    // jitters[ngraph::opset1::Floor::type_info] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::Log::type_info] = CREATE_EMITTER(); // not supported
    jitters[ngraph::opset1::LogicalNot::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_logical_not_emitter);
    jitters[ngraph::opset1::Negative::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_negative_emitter);
    jitters[ngraph::opset1::Relu::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_relu_emitter);
    // jitters[ngraph::opset1::Sign::type_info] = CREATE_EMITTER(); // not supported
    jitters[ngraph::opset1::Sigmoid::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_sigmoid_emitter);
    // jitters[ngraph::opset1::Sin::type_info] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::Sinh::type_info] = CREATE_EMITTER(); // not supported
    jitters[ngraph::opset1::Sqrt::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_sqrt_emitter);
    // jitters[ngraph::opset1::Tan::type_info] = CREATE_EMITTER(); // not supported
    jitters[ngraph::opset1::Tanh::type_info] = CREATE_EMITTER(MKLDNNPlugin::jit_tanh_emitter);

    // jitters[ngraph::opset1::HardSigmoid::type_info] = CREATE_EMITTER(); // not supported
    // jitters[ngraph::opset1::Selu::type_info] = CREATE_EMITTER(); // not supported

    jitters[ngraph::snippets::op::Kernel::type_info] = CREATE_EMITTER(KernelEmitter);
    jitters[ngraph::snippets::op::Tile::type_info] = CREATE_EMITTER(TileEmitter);
}

size_t MKLDNNPlugin::CPUTarget::get_lanes() const {
    switch (isa) {
        case dnnl::impl::cpu::x64::avx2 : return dnnl::impl::cpu::x64::cpu_isa_traits<dnnl::impl::cpu::x64::avx2>::vlen / sizeof(float);
        case dnnl::impl::cpu::x64::sse41 : return dnnl::impl::cpu::x64::cpu_isa_traits<dnnl::impl::cpu::x64::sse41>::vlen / sizeof(float);
        case dnnl::impl::cpu::x64::avx512_common : return dnnl::impl::cpu::x64::cpu_isa_traits<dnnl::impl::cpu::x64::avx512_common>::vlen / sizeof(float);
        default : IE_THROW() << "unknown isa " << isa;
    }
}

bool MKLDNNPlugin::CPUTarget::is_supported() const {
    return dnnl::impl::cpu::x64::mayiuse(isa);
}

code MKLDNNPlugin::CPUTarget::get_snippet() const {
    h->create_kernel();
    return h->jit_ker();
}

MKLDNNPlugin::CPUGenerator::CPUGenerator(dnnl::impl::cpu::x64::cpu_isa_t isa_) : Generator(std::make_shared<CPUTarget>(isa_)) {
}
