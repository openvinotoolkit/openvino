// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_input_node.h"
#include "common/cpu_memcpy.h"
#include "mkldnn_extension_utils.h"

#include <string>
#include <tuple>
#include <algorithm>
#include <utils/general_utils.h>
#include <ngraph/ops.hpp>
#include <ie_parallel.hpp>
#include <ie_ngraph_utils.hpp>
#include <blob_factory.hpp>
#include "caseless.hpp"
#include "common/cpu_memcpy.h"
#include "common/cpu_convert.h"
#include "utils/cpu_utils.hpp"
#include <cpu/x64/jit_generator.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace details;
using namespace ngraph::op;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace {

struct jit_has_subnormals : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_has_subnormals)

    typedef struct {
        const float* src;
        const size_t count;
        bool hasSubnormals;
    } args_t;

    typedef void (*fn_t)(const args_t*);

    jit_has_subnormals() {
        jit_ker_ = nullptr;
    }

    fn_t get() {
        return jit_ker() || create_kernel() == dnnl::impl::status::success
                ? (fn_t)jit_ker()
                : nullptr;
    }

protected:
    void foreach(const Xbyak::Reg64& idx,
                 size_t step,
                 const Xbyak::Reg64& end,
                 std::function<void(const Xbyak::Reg64&)> && fn) {
        Label loop, exit;

        L(loop);
        cmp(idx, end);
        jge(exit);

        fn(idx);

        add(idx, step);
        jmp(loop);
        L(exit);
    }

    void copy_floats(const Xbyak::Reg64& dst,
                     const Xbyak::Reg64& src,
                     const Xbyak::Reg64& size) {
        push(rsi);
        push(r15);

        xor_(rsi, rsi);

        foreach(rsi, 1, size, [&, this](const Xbyak::Reg64& idx) {
            mov(r15d, dword[src + idx * sizeof(float)]);
            mov(dword[dst + idx * sizeof(float)], r15d);
        });

        pop(r15);
        pop(rsi);
    }
};

struct jit_has_subnormals_avx2 : public jit_has_subnormals {
    static const uint32_t vlen = 8u;    // floats vector length for AVX2 instructions

    void check_subnormals(const Xbyak::Reg64& src, const Xbyak::Ymm &mask, const Xbyak::Ymm &zero) {
        auto a = ymm1;
        auto b = ymm2;
        auto c = ymm3;

        vmovdqu(a, yword[src]);         // load 8 floats
        vpcmpeqd(b, a, zero);           // if (a == 0) b = 1 else b = 0
        vpand(c, a, mask);              // c = a & 01111111100000000000000000000000
        vpcmpeqd(c, c, zero);           // if (c == 0) c = 1 else c = 0
        vptest(b, c);                   // if ((!b & c) == 0) CF = 1 else CF = 0
    }

    void generate() final {
        Label exit, has_subnormals, no_subnormals;

        auto reg_src = rax;
        auto reg_dst = rbx;
        auto reg_sz = rdx;
        auto reg_mask_addr = r15;
        auto zero = ymm4;
        auto mask = ymm5;

        preamble();

        // Initialize necessary consts
        vpxor(zero, zero, zero);

        static const uint32_t mask_data[8] = {
            0xFF << 23, 0xFF << 23, 0xFF << 23, 0xFF << 23,
            0xFF << 23, 0xFF << 23, 0xFF << 23, 0xFF << 23
        };

        mov(reg_mask_addr, (size_t)mask_data);
        vmovdqu(mask, yword[reg_mask_addr]);

        // Get arguments addresses
        mov(reg_src, ptr[param1 + offsetof(args_t, src)]);
        lea(reg_dst, ptr[param1 + offsetof(args_t, hasSubnormals)]);
        mov(reg_sz, ptr[param1 + offsetof(args_t, count)]);

        // Main loop
        xor_(rsi, rsi);
        mov(r8, reg_sz);
        shr(r8, 3);

        foreach(rsi, 1, r8, [&, this](const Xbyak::Reg64& idx) {
            check_subnormals(reg_src, mask, zero);
            jnc(has_subnormals);
            add(reg_src, sizeof(float) * vlen);
        });

        // Tail
        shl(rsi, 3);
        sub(reg_sz, rsi);
        test(reg_sz, reg_sz);
        jz(exit);

        // use space on stack for 8 floats
        sub(rsp, vlen * sizeof(float));
        mov(r8, rsp);

        vmovups(yword[r8], zero);

        copy_floats(r8, reg_src, reg_sz);
        check_subnormals(r8, mask, zero);
        jc(no_subnormals);
        add(rsp, vlen * sizeof(float));

        L(has_subnormals);

        mov(rax, 1);
        mov(byte[reg_dst], al);
        jmp(exit);

        L(no_subnormals);
        add(rsp, vlen * sizeof(float));

        L(exit);

        postamble();
    }
};

struct jit_has_subnormals_sse41 : public jit_has_subnormals {
    static const uint32_t vlen = 4u;    // floats vector length for SSE41 instructions

    void check_subnormals(const Xbyak::Reg64& src, const Xbyak::Xmm &mask, const Xbyak::Xmm &zero) {
        auto a = xmm1;
        auto b = xmm2;
        auto c = xmm3;

        movdqu(a, xword[src]);          // load 4 floats
        movdqu(b, a);                   // b = a
        movdqu(c, a);                   // c = a
        pcmpeqd(b, zero);               // if (a == 0) b = 1 else b = 0
        pand(c, mask);                  // c = a & 01111111100000000000000000000000
        pcmpeqd(c, zero);               // if (c == 0) c = 1 else c = 0
        ptest(b, c);                    // if ((!b & c) == 0) CF = 1 else CF = 0
    }

    void generate() final {
        Label exit, has_subnormals, no_subnormals;

        auto reg_src = rax;
        auto reg_dst = rbx;
        auto reg_sz = rdx;
        auto reg_mask_addr = r15;
        auto zero = xmm4;
        auto mask = xmm5;

        preamble();

        // Initialize necessary consts
        pxor(zero, zero);

        static const uint32_t mask_data[4] = {
            0xFF << 23, 0xFF << 23, 0xFF << 23, 0xFF << 23
        };

        mov(reg_mask_addr, (size_t)mask_data);
        movdqu(mask, xword[reg_mask_addr]);

        // Get arguments addresses
        mov(reg_src, ptr[param1 + offsetof(args_t, src)]);
        lea(reg_dst, ptr[param1 + offsetof(args_t, hasSubnormals)]);
        mov(reg_sz, ptr[param1 + offsetof(args_t, count)]);

        // Main loop
        xor_(rsi, rsi);
        mov(r8, reg_sz);
        shr(r8, 2);

        foreach(rsi, 1, r8, [&, this](const Xbyak::Reg64& idx) {
            check_subnormals(reg_src, mask, zero);
            jnc(has_subnormals);
            add(reg_src, sizeof(float) * vlen);
        });

        // Tail
        shl(rsi, 2);
        sub(reg_sz, rsi);
        test(reg_sz, reg_sz);
        jz(exit);

        // use space on stack for 4 floats
        sub(rsp, vlen * sizeof(float));
        mov(r8, rsp);

        movups(xword[r8], zero);

        copy_floats(r8, reg_src, reg_sz);
        check_subnormals(r8, mask, zero);
        jc(no_subnormals);
        add(rsp, vlen * sizeof(float));

        L(has_subnormals);

        mov(rax, 1);
        mov(byte[reg_dst], al);
        jmp(exit);

        L(no_subnormals);
        add(rsp, vlen * sizeof(float));

        L(exit);

        postamble();
    }
};

jit_has_subnormals::fn_t jit_has_subnormals_function() {
    if (mayiuse(cpu_isa_t::avx2)) {
        static jit_has_subnormals_avx2 generator;
        static auto fn = generator.get();
        return fn;
    } else if (mayiuse(cpu_isa_t::sse41)) {
        static jit_has_subnormals_sse41 generator;
        static auto fn = generator.get();
        return fn;
    }
    return nullptr;
}

}   // namespace

MKLDNNInputNode::MKLDNNInputNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(op, eng, cache), origLayer(op) {
    if (!one_of(op->get_type_info(),
            v0::Parameter::type_info,
            v0::Constant::type_info,
            v0::Result::type_info,
            v3::ReadValue::type_info,
            v6::ReadValue::type_info))
        IE_THROW(NotImplemented) << "CPU Input node doesn't support ngraph operation " << op->get_type_name() << " with name " << op->get_friendly_name();

    constant = ConstantType::NoConst;

    auto constOp = ngraph::as_type_ptr<ngraph::op::Constant>(op);
    if (constOp) {
        constant = ConstantType::Const;

        auto dataPrecision = convertPrecision(op->get_element_type());

        size_t shapeSize = ngraph::shape_size(op->get_shape());
        constexpr size_t byte_size{8};
        if (dataPrecision == Precision::BIN) {
            shapeSize = (shapeSize + (byte_size - 1)) / byte_size;
        }

        TensorDesc td(dataPrecision, {shapeSize}, Layout::C);

        constBlob = make_blob_with_precision(td, const_cast<void*>(constOp->get_data_ptr()));

        MKLDNNDims dims(op->get_shape().empty() ? ngraph::Shape(1, 1) : op->get_shape());

        cloneBlobIfRequired(dims, dataPrecision);
     }
}

void MKLDNNInputNode::cloneBlobIfRequired(const MKLDNNDims& dims, const InferenceEngine::Precision& prec) {
    MKLDNNMemoryDesc memDesc(dims, MKLDNNExtensionUtils::IEPrecisionToDataType(prec));

    auto cloneBlob = [&, this] () {
        MKLDNNMemory memory{ getEngine() };
        memory.Create(memDesc, constBlob->buffer());

        MKLDNNMemoryPtr ptr = MKLDNNMemoryPtr(new MKLDNNMemory(getEngine()));
        ptr->Create(memDesc);
        ptr->SetData(memory);

        return ptr;
    };

    auto isBlobAligned = [&, this] () {
        const void *ptr = constBlob->cbuffer().as<const void*>();
        return prec.size() > 1 ? (reinterpret_cast<size_t>(ptr) % prec.size()) == 0 : true;
    };

    // The presence of subnormals is better to determined at IR read time.
    auto hasSubnormals = [&, this] () {
        if (prec == InferenceEngine::Precision::FP32) {
            uint32_t const *u32data = constBlob->cbuffer().as<const uint32_t*>();
            const size_t size = constBlob->byteSize() / prec.size();

            if (!size)
                return false;

            if (auto fn = jit_has_subnormals_function()) {
                static const size_t batch_size = 2048;
                const size_t iterations_num = size / batch_size + 1;

                volatile bool has_subnormals = false;

                parallel_for(iterations_num, [&](int n) {
                    auto ptr = u32data + n * batch_size;
                    const jit_has_subnormals::args_t args = {
                        reinterpret_cast<float const *>(ptr),
                        std::min(batch_size, (size_t)(u32data + size - ptr)),
                        false
                    };

                    fn(&args);

                    if (args.hasSubnormals)
                        has_subnormals = true;
                });

                return has_subnormals;
            } else {
                for (size_t i = 0; i < size; ++i) {
                    if (u32data[i] && (u32data[i] & (0xFF << 23)) == 0) {
                        return true;
                    }
                }
            }
        }
        return false;
    };

    auto blobKey = [this] () {
        char ptr[32];
        snprintf(ptr, sizeof ptr, "%p", constBlob->cbuffer().as<const void*>());
        return getName()
                + "_" + std::to_string(constBlob->byteSize())
                + "_" + ptr;
    };

    const void *data = constBlob->buffer();
    (void)data;

    if (weightCache) {
        memoryPtr = *weightCache->findOrCreate(blobKey(), cloneBlob);
    } else if (isBlobAligned() && !hasSubnormals()) {
        memoryPtr = MKLDNNMemoryPtr(new MKLDNNMemory(getEngine()));
        memoryPtr->Create(memDesc, constBlob->buffer());
    } else {
        memoryPtr = cloneBlob();
    }
}

MKLDNNInputNode::MKLDNNInputNode(const InferenceEngine::SizeVector &dims, const InferenceEngine::Precision &prc, const std::string &name,
                                 const std::string &type, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(type, name, eng, cache) {
    constant = ConstantType::NoConst;
    if (getType() == Input) {
        outDims.emplace_back(dims);
        addOriginalOutputPrecision(prc);
    }  else if (getType() == Output) {
        inDims.emplace_back(dims);
        addOriginalInputPrecision(prc);
    }
}

void MKLDNNInputNode::withMeanImage() {
    isMeanImage = true;
}

const InferenceEngine::Blob::CPtr MKLDNNInputNode::getConstBlob() const {
    return constBlob;
}

MKLDNNMemoryPtr MKLDNNInputNode::getMemoryPtr() const {
    return memoryPtr;
}

void MKLDNNInputNode::getSupportedDescriptors() {
    if (getType() == Input) {
        if (!getParentEdges().empty())
            IE_THROW() << "Incorrect number of input edges for layer " << getName();
        if (getChildEdges().empty())
            IE_THROW() << "Incorrect number of output edges for layer " << getName();
    } else if (getType() == Output) {
        if (getParentEdges().size() != 1)
            IE_THROW() << "Incorrect number of input edges for layer " << getName();
        if (!getChildEdges().empty())
            IE_THROW() << "Incorrect number of output edges for layer " << getName();
    }
}

void MKLDNNInputNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    LayerConfig config;
    config.dynBatchSupport = true;
    if (getType() == Input || getType() == MemoryInput) {
        precision = getOriginalOutputPrecisionAtPort(0);
        if (precision == Precision::U16 || isMeanImage) {
            precision = Precision::FP32;
        }
        DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;

        auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
        auto mem_tdesc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType);
        dataConfig.desc = mem_tdesc;
        config.outConfs.push_back(dataConfig);
        // ReadValue operation expects constant input
        if (!getParentEdges().empty()) {
            DataConfig inConfig;
            inConfig.inPlace = -1;
            inConfig.constant = true;
            inConfig.desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType);
            config.inConfs.push_back(inConfig);
        }
    } else if (getType() == Output) {
        precision = getOriginalInputPrecisionAtPort(0);
        if (precision == Precision::U16) precision = Precision::FP32;
        DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;

        auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
        auto mem_tdesc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType);
        dataConfig.desc = mem_tdesc;
        config.inConfs.push_back(dataConfig);
    }
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
}

void MKLDNNInputNode::createPrimitive() {
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        auto &dstMemPtr = getChildEdgeAt(i)->getMemoryPtr();
        if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
            IE_THROW() << "Destination memory didn't allocate for node " << getName()
                               << " to node " << getChildEdgeAt(i)->getChild()->getName() << ".";
    }
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto &srcMemPtr = getParentEdgeAt(i)->getMemoryPtr();
        if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
            IE_THROW() << "Destination memory didn't allocate for node " << getName()
                               << " from node " << getParentEdgeAt(i)->getParent()->getName() << ".";
    }

    const PrimitiveDescInfo *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set for node " << getName() << ".";
}

bool MKLDNNInputNode::created() const {
    return getType() == Input || getType() == Output;
}

REG_MKLDNN_PRIM_FOR(MKLDNNInputNode, Input);
REG_MKLDNN_PRIM_FOR(MKLDNNInputNode, Output);
