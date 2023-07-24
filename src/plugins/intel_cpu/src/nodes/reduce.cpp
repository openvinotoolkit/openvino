// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce.h"

#include "eltwise.h"
#include "fake_quantize.h"
#include "ie_parallel.hpp"
#include "utils/bfloat16.hpp"
#include <ie_ngraph_utils.hpp>

#include <common/primitive_hashing_utils.hpp>

using namespace ov::intel_cpu::node;
using namespace ov::intel_cpu::kernel;
using namespace InferenceEngine;
using namespace dnnl::impl::cpu;

#define SET_SRC_DIM_VALUE(batch, channel, depth, height, width) IB = batch;   \
                                                                IC = channel; \
                                                                ID = depth;   \
                                                                IH = height;  \
                                                                IW = width;
#define SET_DST_DIM_VALUE(batch, channel, depth, height, width) OB = batch;   \
                                                                OC = channel; \
                                                                OD = depth;   \
                                                                OH = height;  \
                                                                OW = width;

#define GET_OFF(field) offsetof(JitReduceCallArgs, field)
#define GET_OFF_POST(field) offsetof(JitReducePostCallArgs, field)

#define GET_PTR_N_PLN              const uint8_t    *in_ptr_n      = in_ptr       + src_data_size * ib * IC * ID * IH * IW;               \
                                         uint8_t    *out_ptr_n     = out_ptr      + dst_data_size * ob * OC * OD * OH * OW;
#define GET_PTR_NC_PLN             const uint8_t    *in_ptr_nc     = in_ptr_n     + src_data_size * ic * ID * IH * IW;                    \
                                         uint8_t    *out_ptr_nc    = out_ptr_n    + dst_data_size * oc * OD * OH * OW;
#define GET_PTR_NCD_PLN            const uint8_t    *in_ptr_ncd    = in_ptr_nc    + src_data_size * id * IH * IW;                         \
                                         uint8_t    *out_ptr_ncd   = out_ptr_nc   + dst_data_size * od * OH * OW;
#define GET_PTR_NCDH_PLN           const uint8_t    *in_ptr_ncdh   = in_ptr_ncd   + src_data_size * ih * IW;                              \
                                         uint8_t    *out_ptr_ncdh  = out_ptr_ncd  + dst_data_size * oh * OW;
#define GET_PTR_NCD_BASE_PTR_N_PLN const uint8_t    *in_ptr_ncd    = in_ptr_n     + src_data_size * (ic * ID + id) * IH * IW;             \
                                         uint8_t    *out_ptr_ncd   = out_ptr_n    + dst_data_size * (oc * OD + od) * OH * OW;
#define GET_PTR_N_BLK              const uint8_t    *in_ptr_n      = in_ptr       + src_data_size * ib * ICB * ID * IH * IW * blk_size;   \
                                         uint8_t    *out_ptr_n     = out_ptr      + dst_data_size * ob * OCB * OD * OH * OW * blk_size;
#define GET_PTR_NC_BLK             const uint8_t    *in_ptr_nc     = in_ptr_n     + src_data_size * icb * ID * IH * IW * blk_size;        \
                                         uint8_t    *out_ptr_nc    = out_ptr_n    + dst_data_size * ocb * OD * OH * OW * blk_size;
#define GET_PTR_NCD_BLK            const uint8_t    *in_ptr_ncd    = in_ptr_nc    + src_data_size * id * IH * IW * blk_size;              \
                                         uint8_t    *out_ptr_ncd   = out_ptr_nc   + dst_data_size * od * OH * OW * blk_size;
#define GET_PTR_NCDH_BLK           const uint8_t    *in_ptr_ncdh   = in_ptr_ncd   + src_data_size * ih * IW * blk_size;                   \
                                         uint8_t    *out_ptr_ncdh  = out_ptr_ncd  + dst_data_size * oh * OW * blk_size;
#define GET_PTR_NCDHW_BLK          const uint8_t    *in_ptr_ncdhw  = in_ptr_ncdh  + src_data_size * iw * blk_size;                        \
                                         uint8_t    *out_ptr_ncdhw = out_ptr_ncdh + dst_data_size * ow * blk_size;
#define GET_PTR_NCD_BASE_PTR_N_BLK const uint8_t    *in_ptr_ncd    = in_ptr_n     + src_data_size * (icb * ID + id) * IH * IW * blk_size; \
                                         uint8_t    *out_ptr_ncd   = out_ptr_n    + dst_data_size * (ocb * OD + od) * OH * OW * blk_size;

namespace {

struct ReduceKey {
    JitReduceConfigParams jcp;
    dnnl::post_ops postOps;

    size_t hash() const;
    bool operator==(const ReduceKey& rhs) const;
};

size_t ReduceKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    seed = hash_combine(seed, jcp.layout);
    seed = hash_combine(seed, jcp.reduce_mode);
    seed = hash_combine(seed, jcp.fuse_low_precision);
    seed = hash_combine(seed, (ov::element::Type_t)jcp.src_el_type);
    seed = hash_combine(seed, (ov::element::Type_t)jcp.dst_el_type);
    seed = get_post_op_hash(seed, *postOps.get());

    return seed;
}

bool ReduceKey::operator==(const ReduceKey &rhs) const {
    return jcp.layout == rhs.jcp.layout && jcp.reduce_mode == rhs.jcp.reduce_mode &&
           jcp.fuse_low_precision == rhs.jcp.fuse_low_precision &&
           jcp.src_el_type == rhs.jcp.src_el_type && jcp.dst_el_type == rhs.jcp.dst_el_type && *postOps.get() == *rhs.postOps.get();
}

} // namespace

const std::map<const ov::DiscreteTypeInfo, std::function<void(const std::shared_ptr<ov::Node>&, Reduce&)>> Reduce::initializers = {
    {op::v4::ReduceL1::get_type_info_static(), [](const std::shared_ptr<ov::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceL1;
    }},
    {op::v4::ReduceL2::get_type_info_static(), [](const std::shared_ptr<ov::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceL2;
    }},
    {op::v1::ReduceLogicalAnd::get_type_info_static(), [](const std::shared_ptr<ov::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceAnd;
    }},
    {op::v1::ReduceLogicalOr::get_type_info_static(), [](const std::shared_ptr<ov::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceOr;
    }},
    {op::v1::ReduceMax::get_type_info_static(), [](const std::shared_ptr<ov::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceMax;
    }},
    {op::v1::ReduceMean::get_type_info_static(), [](const std::shared_ptr<ov::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceMean;
    }},
    {op::v1::ReduceMin::get_type_info_static(), [](const std::shared_ptr<ov::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceMin;
    }},
    {op::v1::ReduceProd::get_type_info_static(), [](const std::shared_ptr<ov::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceProd;
    }},
    {op::v1::ReduceSum::get_type_info_static(), [](const std::shared_ptr<ov::Node>& op, Reduce& node) {
        node.algorithm = Algorithm::ReduceSum;
    }}
};

bool Reduce::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!op->get_type_info().is_castable(op::util::ArithmeticReductionKeepDims::get_type_info_static()) &&
                !op->get_type_info().is_castable(op::util::LogicalReductionKeepDims::get_type_info_static())) {
            errorMessage = "Reduce node with name " + op->get_friendly_name() + " is not derived from ArithmeticReductionKeepDims or LogicalReductionKeepDims";
            return false;
        }
        const auto idxIn = op->get_input_node_shared_ptr(REDUCE_INDEXES);
        if (idxIn->get_type_info() != op::v0::Constant::get_type_info_static()) {
            errorMessage = "Only const 'reduce_indexes' input is supported";
            return false;
        }
        if (idxIn->get_element_type() != ov::element::i32 && idxIn->get_element_type() != ov::element::i64) {
            errorMessage = "Only i32 and i64 'reduce_indexes' input is supported";
            return false;
        }
        if (initializers.find(op->get_type_info()) == initializers.end()) {
            errorMessage = "Doesn't support Reduce algorithm: " +  std::string(op->get_type_info().name);
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Reduce::Reduce(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, NgraphShapeInferFactory(op, PortMask(REDUCE_INDEXES))) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    initializers.at(op->get_type_info())(op, *this);

    if (const auto reduction = std::dynamic_pointer_cast<op::util::ReductionBase>(op)) {
        keep_dims = reduction->get_keep_dims();
    }
    const auto idxIn = ov::as_type_ptr<op::v0::Constant>(op->get_input_node_shared_ptr(REDUCE_INDEXES));
    if (idxIn->get_element_type() == ov::element::i32) {
        const auto tmpData = idxIn->get_vector<int32_t>();
        raw_axes.assign(tmpData.begin(), tmpData.end());
    } else if (idxIn->get_element_type() == ov::element::i64) {
        raw_axes = idxIn->get_vector<int64_t>();
    }

    set_use_aux_kernel = false;
    fuse_low_precision = false;
    vec_reduceDH_prc.clear();
    vec_reduceCDW_prc.clear();
    setJITBeyond5D();
}

void Reduce::getSupportedDescriptors() {
    if (getParentEdges().size() != 2) {
        THROW_CPU_NODE_ERR << " gets incorrect number of input edges!";
    }
    if (getChildEdges().empty()) {
        THROW_CPU_NODE_ERR << " gets incorrect number of output edges!";
    }

    if (getInputShapeAtPort(REDUCE_INDEXES).getRank() != 1) {
        THROW_CPU_NODE_ERR << " gets incorrect index vector dimension! Index vector should be 1 dimension.";
    }

    if (keep_dims) {
        if (getInputShapeAtPort(REDUCE_DATA).getRank() != getOutputShapeAtPort(0).getRank())
            THROW_CPU_NODE_ERR << " gets incorrect number of input/output dimensions!";
    } else {
        // In fact, after the Reduce operation, the shape must be a scalar if the previous one was 1d.
        // But for now, 0d tensor (scalar) is emulated as 1d tensor. Skip checking in such cases.
        bool is_emulated_0d_as_1d = getInputShapeAtPort(REDUCE_DATA).getRank() == 1 && getOutputShapeAtPort(0).getRank() == 1;
        if (getInputShapeAtPort(REDUCE_DATA).getRank() <= getOutputShapeAtPort(0).getRank() && !is_emulated_0d_as_1d)
            THROW_CPU_NODE_ERR << "gets incorrect number of input/output dimensions!";
    }
}

void Reduce::initSupportedPrimitiveDescriptors() {
    const auto& input_prec_0 = getOriginalInputPrecisionAtPort(REDUCE_DATA);
    auto input_prec_1 = getOriginalInputPrecisionAtPort(REDUCE_INDEXES);
    output_prec = getOriginalOutputPrecisionAtPort(0);

    if (input_prec_1 == Precision::U64) {
        input_prec_1 = Precision::I64;
    } else if (!one_of(input_prec_1, Precision::I32, Precision::I64)) {
        input_prec_1 = Precision::I32;
    }

    if (!fusedWith.empty()) {
        // In jit mode we use the output memory as an intermediate accumulator for certain reduce modes.
        // If the post ops node has a lower precision for such modes, working buffer with original precision is needed,
        // in order to avoid accuracy loss.
        auto fused_prec = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
        if (output_prec == Precision::FP32 && fused_prec != Precision::FP32) {
            if (algorithm != Algorithm::ReduceAnd && algorithm != Algorithm::ReduceOr &&
                algorithm != Algorithm::ReduceMin && algorithm != Algorithm::ReduceMax) {
                fuse_low_precision = true;
            }
        }
        output_prec = fused_prec;
    }

    jit_mode = canApplyJIT(input_prec_0, output_prec);

    if (jit_mode) {
        // Since in jit mode we use the output memory as an intermediate accumulator for certain reduce modes, we can't use BF16 output precision due to
        // the possible accuracy loss. Therefore, for such mods, we will change the output precision to FP32.
        if (Precision::BF16 == output_prec) {
            if (!x64::mayiuse(x64::avx512_core)) {
                output_prec = Precision::FP32;
            } else if (algorithm != Algorithm::ReduceAnd && algorithm != Algorithm::ReduceOr &&
                       algorithm != Algorithm::ReduceMin && algorithm != Algorithm::ReduceMax) {
                output_prec = Precision::FP32;
            }
        }
    }

    intermediate_prec = fuse_low_precision ? Precision(Precision::FP32) : output_prec;
    precision_change = input_prec_0 != intermediate_prec;
    support_split = algorithm != Algorithm::ReduceL2 && algorithm != Algorithm::ReduceLogSumExp &&
                    algorithm != Algorithm::ReduceSumSquare;

    src_data_size = input_prec_0.size();
    dst_data_size = output_prec.size();
    intermediate_data_size = intermediate_prec.size();

    NodeConfig config;
    config.inConfs.resize(2);
    config.outConfs.resize(1);
    config.inConfs[REDUCE_DATA].constant(false);
    config.inConfs[REDUCE_INDEXES].constant(false);
    config.outConfs[0].constant(false);
    config.inConfs[REDUCE_DATA].inPlace(-1);
    config.inConfs[REDUCE_INDEXES].inPlace(-1);
    config.outConfs[0].inPlace(-1);

    auto& creatorsMap = BlockedDescCreator::getCommonCreators();

    auto pushDesc = [&](const LayoutType &inFormat, const LayoutType &outFormat, const Precision& inPrecision0, const Precision& inPrecision1,
                        const Precision& outPrecision, const impl_desc_type &impl_type, bool useAclExecutor = false) {
        config.inConfs[REDUCE_DATA].setMemDesc(creatorsMap.at(inFormat)->createSharedDesc(inPrecision0, getInputShapeAtPort(REDUCE_DATA)));
        config.inConfs[REDUCE_INDEXES].setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(inPrecision1,
                                                                                                     getInputShapeAtPort(REDUCE_INDEXES)));
        config.outConfs[0].setMemDesc(creatorsMap.at(outFormat)->createSharedDesc(outPrecision, getOutputShapeAtPort(0)));

        if (useAclExecutor) {
            std::vector<MemoryDescPtr> srcMemoryDescs;
            for (size_t i = 0; i < config.inConfs.size(); i++) {
                srcMemoryDescs.push_back(config.inConfs[i].getMemDesc());
            }
            std::vector<MemoryDescPtr> dstMemoryDescs;
            for (size_t i = 0; i < config.outConfs.size(); i++) {
                dstMemoryDescs.push_back(config.outConfs[i].getMemDesc());
            }

            auto factory = std::make_shared<ReduceExecutorFactory>(reduceAttrs, srcMemoryDescs, dstMemoryDescs,
                                                                   std::make_shared<ExecutorContext>(context, getImplPriority()));
            if (!factory->isEmpty()) {
                supportedPrimitiveDescriptors.push_back({config, impl_type, factory});
            }
        } else {
            supportedPrimitiveDescriptors.push_back({config, impl_type});
        }
    };

#if defined (OV_CPU_WITH_ACL)
        reduceAttrs.operation = algorithm;
        reduceAttrs.keepDims = keep_dims;
        reduceAttrs.axes = raw_axes;
        for (auto &axis : reduceAttrs.axes) {
            if (axis < 0)
                axis += static_cast<int>(getInputShapeAtPort(REDUCE_DATA).getRank());
        }
        // TODO: Per-channel layout is disabled due to accuracy issue in ACL Reduce Executor
        // pushDesc(LayoutType::nspc, LayoutType::nspc, input_prec, output_prec, undef, true);
        pushDesc(LayoutType::ncsp, LayoutType::ncsp, input_prec, output_prec, impl_desc_type::undef, true);
        canUseAclExecutor = !supportedPrimitiveDescriptors.empty();
        if (canUseAclExecutor)
            return;
#endif

    if (jit_mode) {
        impl_desc_type impl_type = impl_desc_type::jit_sse42;
        if (x64::mayiuse(x64::avx512_core)) {
            impl_type = impl_desc_type::jit_avx512;
        } else if (x64::mayiuse(x64::avx2)) {
            impl_type = impl_desc_type::jit_avx2;
        }

        pushDesc(LayoutType::ncsp, LayoutType::ncsp, input_prec_0, input_prec_1, output_prec, impl_type);
        if ((getInputShapeAtPort(REDUCE_DATA).getRank() == 4 || getInputShapeAtPort(REDUCE_DATA).getRank() == 5) &&
                getInputShapeAtPort(REDUCE_DATA).getMinDims()[1] > 1) {
            if (keep_dims) {
                pushDesc(LayoutType::nspc, LayoutType::nspc, input_prec_0, input_prec_1, output_prec, impl_type);
                if (x64::mayiuse(x64::avx512_core)) {
                    if (src_data_size <= 4) {
                        pushDesc(LayoutType::nCsp16c, LayoutType::nCsp16c, input_prec_0, input_prec_1, output_prec, impl_type);
                    } else if (src_data_size == 8) {
                        pushDesc(LayoutType::nCsp8c, LayoutType::nCsp8c, input_prec_0, input_prec_1, output_prec, impl_type);
                    }
                } else if (src_data_size <= 4) {
                    pushDesc(LayoutType::nCsp8c, LayoutType::nCsp8c, input_prec_0, input_prec_1, output_prec, impl_type);
                }
            } else {
                pushDesc(LayoutType::nspc, LayoutType::ncsp, input_prec_0, input_prec_1, output_prec, impl_type);
                if (x64::mayiuse(x64::avx512_core)) {
                    if (src_data_size <= 4) {
                        pushDesc(LayoutType::nCsp16c, LayoutType::ncsp, input_prec_0, input_prec_1, output_prec, impl_type);
                    } else if (src_data_size == 8) {
                        pushDesc(LayoutType::nCsp8c, LayoutType::ncsp, input_prec_0, input_prec_1, output_prec, impl_type);
                    }
                } else if (src_data_size <= 4) {
                    pushDesc(LayoutType::nCsp8c, LayoutType::ncsp, input_prec_0, input_prec_1, output_prec, impl_type);
                }
            }
        }
    } else {
        pushDesc(LayoutType::ncsp, LayoutType::ncsp, Precision::FP32, Precision::I32, Precision::FP32, impl_desc_type::ref);
    }
}

bool Reduce::isExecutable() const {
    return !isInputTensorAtPortEmpty(REDUCE_DATA);
}

void Reduce::prepareParams() {
    if (canUseAclExecutor) {
        std::vector<MemoryDescPtr> srcMemoryDescs;
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            srcMemoryDescs.push_back(getParentEdgeAt(i)->getMemoryPtr()->getDescPtr());
        }
        std::vector<MemoryDescPtr> dstMemoryDescs;
        dstMemoryDescs.push_back(getChildEdgeAt(0)->getMemoryPtr()->getDescPtr());

        auto selectedPD = getSelectedPrimitiveDescriptor();
        aclExecPtr = selectedPD->getExecutorFactoryAs<ReduceExecutorFactory>()->makeExecutor(reduceAttrs, srcMemoryDescs, dstMemoryDescs, {});
        selectedPD->setImplementationType(aclExecPtr->getImplType());

        return;
    }

    src_dims = getParentEdgesAtPort(REDUCE_DATA)[0]->getMemory().getDesc().getShape().getDims();
    std::vector<int64_t> reduce_axes;
    if (jit_mode && jit_beyond_5D) {
        reduce_axes = update_src_dims();
    } else {
        reduce_axes = raw_axes;
    }

    auto dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    const SizeVector &dst_dims = dstMemPtr->getDesc().getShape().getDims();
    dst_size = dstMemPtr->getSize();
    calc_process_dst_dims(reduce_axes, dst_dims);
    if (jit_mode) {
        set_reduce_dim_flags();
    }

    apply_post_kernel = true;
    apply_division = false;

    auto builder = [&](const ReduceKey& key) -> std::shared_ptr<JitReduceKernelBase<JitReducePostCallArgs>> {
        std::shared_ptr<JitReduceKernelBase<JitReducePostCallArgs>> postKernel;
#if defined(OPENVINO_ARCH_X86_64)
        if (x64::mayiuse(x64::avx512_core)) {
            postKernel.reset(new JitReducePostKernel<x64::avx512_core>(key.jcp, *attr.get()));
        } else if (x64::mayiuse(x64::avx2)) {
            postKernel.reset(new JitReducePostKernel<x64::avx2>(key.jcp, *attr.get()));
        } else if (x64::mayiuse(x64::sse41)) {
            postKernel.reset(new JitReducePostKernel<x64::sse41>(key.jcp, *attr.get()));
        }
#endif // OPENVINO_ARCH_X86_64
        if (postKernel) {
            postKernel->create_kernel();
        }

        return postKernel;
    };

    if (compile_post_kernel) {
        setPostOps(attr, dst_dims, true);

        ReduceKey key = {jcp, attr.get_post_ops()};
        auto cache = context->getParamsCache();
        auto result = cache->getOrCreate(key, builder);
        if (!result.first) {
            THROW_CPU_NODE_ERR << " has not found JitReducePostKernel.";
        }

        reduce_post_kernel = result.first;
        jit_mode = jit_mode && reduce_post_kernel;

        if (jit_mode) {
            size_t divisor = IB * IC * ID * IH * IW / (OB * OC * OD * OH * OW);
            if (divisor == 0lu) {
                divisor = 1lu;
            }
            if (reduce_post_kernel->get_exec_prc().size() == 4) {
                in_out_divisor_f32 = static_cast<float>(divisor);
                in_out_divisor = &in_out_divisor_f32;
            } else if (reduce_post_kernel->get_exec_prc().size() == 8) {
                in_out_divisor_f64 = static_cast<double>(divisor);
                in_out_divisor = &in_out_divisor_f64;
            }
        }

        if (!isDynamicNode()) {
            compile_post_kernel = false;
        }
    }
}

void Reduce::createPrimitive() {
    if (!isExecutable()) {
        return;
    }
    auto dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto srcMemPtr = getParentEdgeAt(REDUCE_DATA)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        THROW_CPU_NODE_ERR << " has not allocated destination memory.";
    if (!srcMemPtr || !srcMemPtr->isAllocated())
        THROW_CPU_NODE_ERR << " has not allocate input memory.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_CPU_NODE_ERR << " has nullable preferable primitive descriptor";

    if (srcMemPtr->getDesc().hasLayoutType(LayoutType::ncsp)) {
        layout = ReduceLayoutType::reduce_ncsp;
    } else if (srcMemPtr->getDesc().hasLayoutType(LayoutType::nspc)) {
        layout = ReduceLayoutType::reduce_nspc;
    } else {
        layout = ReduceLayoutType::reduce_blocked;
    }

    // hybrid layout: nspc/blocked layout for input and ncsp for output
    // !keepDims is needed to avoid hybrid layout for cases eg. (A, B, C, D) reduce to (A, 1, 1, 1)
    if (!keep_dims && (layout == ReduceLayoutType::reduce_nspc || layout == ReduceLayoutType::reduce_blocked)) {
        is_hybrid_layout = dstMemPtr->getDesc().hasLayoutType(LayoutType::ncsp);
    }

    auto selectedPD = getSelectedPrimitiveDescriptor();
    jcp = JitReduceConfigParams();
    jcp.src_el_type = details::convertPrecision(selectedPD->getConfig().inConfs[REDUCE_DATA].getMemDesc()->getPrecision());
    jcp.dst_el_type = details::convertPrecision(selectedPD->getConfig().outConfs[0].getMemDesc()->getPrecision());
    jcp.layout = layout;
    jcp.reduce_mode = getAlgorithm();
    jcp.fuse_low_precision = fuse_low_precision;

#if defined(OPENVINO_ARCH_X86_64)
    compile_post_kernel = true;
#else
    compile_post_kernel = false;
#endif // OPENVINO_ARCH_X86_64

    size_t prcDiv = jcp.src_el_type.size() < 4 ? 4 : jcp.src_el_type.size();
    if (x64::mayiuse(x64::avx512_core)) {
        blk_size = 64 / prcDiv;
    } else {
        blk_size = 32 / prcDiv;
    }

    if (inputShapesDefined()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }

    auto reduce_jcp = jcp;
    reduce_jcp.dst_el_type = fuse_low_precision ? details::convertPrecision(intermediate_prec) : jcp.dst_el_type;
    create_reduce_kernel(reduce_kernel, reduce_jcp);

    // set_use_aux_kernel being false means this is a dynamic case, and prepareParams() hasn't been invoked yet.
    // So set use_aux_kernel true if precision changes, in case ReduceDH_opt, ReduceCDW_opt or ReduceAll_opt
    // should be true when invoking prepareParams(), then aux kernel will be needed.
    if (!set_use_aux_kernel) {
        use_aux_kernel = precision_change;
        set_use_aux_kernel = true;
    }

    // For scenarios(e.g. when ReduceDH_opt or ReduceAll_opt is true) that apply two stages of kernel invocation
    // to improve parallelism, if the precision is asymmetrical, we apply the aux kernel on the second stage. For
    // example, if the original kernel is bf16-in-fp32-out, then this original kernel will be applied on first
    // stage to reduce some dimensions, and an extra fp32-in-fp32-out aux kernel will be applied on the second
    // stage to reduce the rest dimensions.
    if (use_aux_kernel) {
        aux_jcp = reduce_jcp;
        aux_jcp.src_el_type = reduce_jcp.dst_el_type;
        create_reduce_kernel(reduce_aux_kernel, aux_jcp);
    }
}

void Reduce::create_reduce_kernel(std::shared_ptr<JitReduceKernelBase<kernel::JitReduceCallArgs>> &kernel, const JitReduceConfigParams &jcp) {
#if defined(OPENVINO_ARCH_X86_64)
    if (x64::mayiuse(x64::avx512_core)) {
        kernel.reset(new JitReduceKernel<x64::avx512_core>(jcp));
    } else if (x64::mayiuse(x64::avx2)) {
        kernel.reset(new JitReduceKernel<x64::avx2>(jcp));
    } else if (x64::mayiuse(x64::sse41)) {
        kernel.reset(new JitReduceKernel<x64::sse41>(jcp));
    }
#endif // OPENVINO_ARCH_X86_64
    if (kernel) {
        kernel->create_kernel();
    }
    jit_mode = jit_mode && kernel;
}

void Reduce::execute(dnnl::stream strm) {
    auto dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto srcMemPtr = getParentEdgeAt(REDUCE_DATA)->getMemoryPtr();

    const uint8_t *src_data = reinterpret_cast<const uint8_t *>(srcMemPtr->getData());
    uint8_t *dst_data = reinterpret_cast<uint8_t *>(dstMemPtr->getData());

    if (jit_mode) {
        if (is_hybrid_layout) {
            dst_data = reinterpret_cast<uint8_t *>(prc_mem.get_data_handle());
        }
        reduce_type(src_data, dst_data);
    } else if (aclExecPtr) {
        std::vector<MemoryCPtr> srcMemory;
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            srcMemory.push_back(getParentEdgeAt(i)->getMemoryPtr());
        }
        std::vector<MemoryPtr> dstMemory;
        dstMemory.push_back(getChildEdgeAt(0)->getMemoryPtr());

        aclExecPtr->exec(srcMemory, dstMemory, postOpsDataPtrs.data());
    } else {
        if (layout == ReduceLayoutType::reduce_ncsp) {
            auto in_ptr = reinterpret_cast<const float *>(src_data);
            auto out_ptr = reinterpret_cast<float *>(dst_data);
            reduce_ref(in_ptr, out_ptr);
        } else {
            THROW_CPU_NODE_ERR << " supports only plain layout on machine w/o sse42.";
        }
    }
}

void Reduce::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void Reduce::reduce_type(const uint8_t *in_ptr, uint8_t *out_ptr) {
    reduce_stride = IW;

    if (layout == ReduceLayoutType::reduce_ncsp || layout == ReduceLayoutType::reduce_nspc) {
        reduce_PLN(in_ptr, out_ptr);
    } else {
        if (ReduceC && (IC % blk_size)) {
            reduce_BLK_concern_padding(in_ptr, out_ptr);
        } else {
            reduce_BLK(in_ptr, out_ptr);
        }
    }

    if (is_hybrid_layout) {
        uint8_t *proc_ptr = out_ptr;
        auto dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
        out_ptr = reinterpret_cast<uint8_t *>(dstMemPtr->getData());
        if (layout == ReduceLayoutType::reduce_nspc) {
            switch (dst_data_size) {
                case 1: nspc2ncsp<uint8_t>(proc_ptr, out_ptr); break;
                case 2: nspc2ncsp<uint16_t>(proc_ptr, out_ptr); break;
                case 4: nspc2ncsp<uint32_t>(proc_ptr, out_ptr); break;
                case 8: nspc2ncsp<uint64_t>(proc_ptr, out_ptr); break;
            }
        } else {
            switch (dst_data_size) {
                case 1: blocked2ncsp<uint8_t>(proc_ptr, out_ptr); break;
                case 2: blocked2ncsp<uint16_t>(proc_ptr, out_ptr); break;
                case 4: blocked2ncsp<uint32_t>(proc_ptr, out_ptr); break;
                case 8: blocked2ncsp<uint64_t>(proc_ptr, out_ptr); break;
            }
        }
    }
}

void Reduce::reduce_PLN(const uint8_t *in_ptr, uint8_t *out_ptr) {
    output_info_reassign(out_ptr);
    init_dst_data(out_ptr, dst_size);

    if (ReduceN && !ReduceC && !ReduceD && !ReduceH && !ReduceW) {
        size_t IA = IC * ID * IH * IW;
        reduce_stride = IA;
        parallel_for(IA / blk_size, [&](size_t iba){
            size_t oba = iba;
            reduce_kernel_process(in_ptr + iba * blk_size * src_data_size, out_ptr + oba * blk_size * dst_data_size,
                                  blk_size, 0, IB);
        });

        size_t tail_start = IA / blk_size * blk_size;
        reduce_kernel_process(in_ptr + tail_start * src_data_size, out_ptr + tail_start * dst_data_size,
                              IA - tail_start, 0, IB);
    } else {
        for (size_t ib = 0; ib < IB; ib++) {
            size_t ob = ReduceN ? 0 : ib; GET_PTR_N_PLN;
            if (!ReduceC && !ReduceD && ReduceW) {
                size_t work_amount = ReduceH ? IH * IW : IW;
                if (work_amount < blk_size && x64::mayiuse(x64::avx2)) {
                    size_t outer_size = ReduceH ? IC * ID : IC * ID * IH;
                    size_t inner_size = ReduceH ? IH * IW : IW;
                    size_t output_inner_size = ReduceH ? OH * OW : OW;
                    size_t IK = outer_size / blk_size;
                    std::vector<int> indicesBuf(16, work_amount * src_data_size);
                    for (size_t i = 0; i < blk_size; i++) {
                        indicesBuf[i] *= i;
                    }
                    parallel_for(IK, [&](size_t ik) {
                        size_t ok = ik;
                        reduce_kernel_process(in_ptr_n + ik * blk_size * inner_size * src_data_size,
                                              out_ptr_n + ok * blk_size * output_inner_size * dst_data_size,
                                              work_amount, 1, 0, static_cast<int *>(&indicesBuf[0]));
                    });
                    size_t tail_start = IK * blk_size;
                    size_t IT = outer_size - tail_start;
                    parallel_for(IT, [&](size_t it) {
                        size_t ot = it;
                        reduce_kernel_process(in_ptr_n + (tail_start + it) * inner_size * src_data_size,
                                              out_ptr_n + (tail_start + ot) * output_inner_size * dst_data_size, work_amount, 1);
                    });
                } else {
                    if (ReduceH) {
                        parallel_for2d(IC, ID, [&](size_t ic, size_t id) {
                            size_t oc = ic, od = id; GET_PTR_NCD_BASE_PTR_N_PLN;
                            reduce_kernel_process(in_ptr_ncd, out_ptr_ncd, work_amount, 1);
                        });
                    } else {
                        parallel_for3d(IC, ID, IH, [&](size_t ic, size_t id, size_t ih) {
                            size_t oc = ic, od = id; GET_PTR_NCD_BASE_PTR_N_PLN;
                            size_t oh = ih; GET_PTR_NCDH_PLN;
                            reduce_kernel_process(in_ptr_ncdh, out_ptr_ncdh, work_amount, 1);
                        });
                    }
                }
            } else if (ReduceH && ReduceW) {
                for (size_t ic = 0; ic < IC; ic++) {
                    size_t oc = ReduceC ? 0 : ic; GET_PTR_NC_PLN;
                    for (size_t id = 0; id < ID; id++) {
                        size_t od = ReduceD ? 0 : id; GET_PTR_NCD_PLN;
                        reduce_kernel_process(in_ptr_ncd, out_ptr_ncd, IH * IW, 1);
                    }
                }
            } else if (!ReduceH && ReduceW) {
                if (ReduceCDW_opt) {
                    // reduce parallelly in HW dimensions
                    // step1: ReduceC && ReduceD && !ReduceH && !ReduceW
                    uint8_t *prc_ptr_n = &vec_reduceCDW_prc[0];
                    init_dst_data(prc_ptr_n, prc_size);
                    size_t IS = IH * IW;
                    reduce_stride = IS;
                    parallel_for(IS / blk_size, [&](size_t ibs){
                        size_t pbs = ibs;
                        reduce_kernel_process(in_ptr_n + ibs * blk_size * src_data_size, prc_ptr_n + pbs * blk_size * prc_data_size,
                                              blk_size, 0, IC * ID);
                    });
                    size_t tail_start = IS / blk_size * blk_size;
                    reduce_kernel_process(in_ptr_n + tail_start * src_data_size, prc_ptr_n + tail_start * prc_data_size,
                                          IS - tail_start, 0, IC * ID);
                    // step2: ReduceW
                    reduce_kernel_reassign();
                    parallel_for(PH, [&](size_t ph){
                        size_t oh = ph;
                        reduce_kernel_process(prc_ptr_n + ph * PW * prc_data_size, out_ptr_n + oh * OW * dst_data_size, IW, 1);
                    });
                    reduce_kernel_restore();
                } else {
                    for (size_t ic = 0; ic < IC; ic++) {
                        size_t oc = ReduceC ? 0 : ic; GET_PTR_NC_PLN;
                        for (size_t id = 0; id < ID; id++) {
                            size_t od = ReduceD ? 0 : id; GET_PTR_NCD_PLN;
                            parallel_for(IH, [&](size_t ih){
                                size_t oh = ih; GET_PTR_NCDH_PLN;
                                reduce_kernel_process(in_ptr_ncdh, out_ptr_ncdh, IW, 1);
                            });
                        }
                    }
                }
            } else if (ReduceW) {
                for (size_t ic = 0; ic < IC; ic++) {
                    size_t oc = ReduceC ? 0 : ic; GET_PTR_NC_PLN;
                    for (size_t id = 0; id < ID; id++) {
                        size_t od = ReduceD ? 0 : id; GET_PTR_NCD_PLN;
                        for (size_t ih = 0; ih < IH; ih++) {
                            size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_PLN;
                            reduce_kernel_process(in_ptr_ncdh, out_ptr_ncdh, IW, 1);
                        }
                    }
                }
            } else if (!ReduceC && !ReduceD && ReduceH && !ReduceW) {
                parallel_for2d(IC, ID, [&](size_t ic, size_t id) {
                    size_t oc = ic, od = id;
                    GET_PTR_NCD_BASE_PTR_N_PLN;
                    parallel_for(IW / blk_size, [&](size_t ibw) {
                        size_t obw = ibw;
                        reduce_kernel_process(in_ptr_ncd + ibw * blk_size * src_data_size,
                                            out_ptr_ncd + obw * blk_size * dst_data_size,
                                            blk_size, 0, IH);
                    });
                    size_t tail_start = IW / blk_size * blk_size;
                    reduce_kernel_process(in_ptr_ncd + tail_start * src_data_size,
                                        out_ptr_ncd + tail_start * dst_data_size,
                                        IW - tail_start, 0, IH);
                });
            } else if (!ReduceC && ReduceD && ReduceH && !ReduceW) {
                size_t IWB = IW / blk_size;
                if (ReduceDH_opt) {
                    if (IWB > 0) {
                        // reduce parallelly in D dimension
                        // step1: !ReduceD && ReduceH && !ReduceW
                        uint8_t *prc_ptr_n = &vec_reduceDH_prc[0];
                        init_dst_data(prc_ptr_n, prc_size);
                        parallel_for2d(ID, IWB, [&](size_t id, size_t iwb) {
                            size_t pd = id, pwb = iwb;
                            reduce_kernel_process(in_ptr_n + (id * IH * IW + iwb * blk_size) * src_data_size,
                                                  prc_ptr_n + (pd * PW + pwb * blk_size) * prc_data_size, blk_size, 0, IH);
                        });
                        // step2: ReduceD
                        reduce_stride = PW;
                        reduce_kernel_reassign();
                        parallel_for(IWB, [&](size_t iwb){
                            size_t pwb = iwb, owb = iwb;
                            reduce_kernel_process(prc_ptr_n + pwb * blk_size * prc_data_size,
                                                out_ptr_n + owb * blk_size * dst_data_size, blk_size, 0, ID);
                        });
                        reduce_kernel_restore();
                    }
                    // reduce tail
                    reduce_stride = IW;
                    size_t tail_start = IWB * blk_size;
                    parallel_for(IW - tail_start, [&](size_t i_tail) {
                        reduce_kernel_process(in_ptr_n + (tail_start + i_tail) * src_data_size, out_ptr_n + (tail_start + i_tail) * dst_data_size,
                                              1, 0, ID * IH);
                    });
                } else {
                    parallel_for(IC, [&](size_t ic) {
                        size_t oc = ic; GET_PTR_NC_PLN;
                        parallel_for(IWB, [&](size_t iwb){
                            size_t owb = iwb;
                            reduce_kernel_process(in_ptr_nc + iwb * blk_size * src_data_size, out_ptr_nc + owb * blk_size * dst_data_size,
                                                blk_size, 0, ID * IH);
                        });
                        size_t tail_start = IWB * blk_size;
                        parallel_for(IW - tail_start, [&](size_t i_tail) {
                            reduce_kernel_process(in_ptr_nc + (tail_start + i_tail) * src_data_size, out_ptr_nc + (tail_start + i_tail) * dst_data_size,
                                                1, 0, ID * IH);
                        });
                    });
                }
            } else if (ReduceC && ReduceD && ReduceH && !ReduceW) {
                parallel_for(IW / blk_size, [&](size_t ibw){
                    size_t obw = ibw;
                    reduce_kernel_process(in_ptr_n + ibw * blk_size * src_data_size, out_ptr_n + obw * blk_size * dst_data_size,
                                          blk_size, 0, IC * ID * IH);
                });

                size_t tail_start = IW / blk_size * blk_size;
                reduce_kernel_process(in_ptr_n + tail_start * src_data_size, out_ptr_n + tail_start * dst_data_size,
                                      IW - tail_start, 0, IC * ID * IH);
            } else if (ReduceC && !ReduceD && !ReduceH && !ReduceW) {
                size_t IS = ID * IH * IW;
                reduce_stride = IS;
                parallel_for(IS / blk_size, [&](size_t ibs){
                    size_t obs = ibs;
                    reduce_kernel_process(in_ptr_n + ibs * blk_size * src_data_size, out_ptr_n + obs * blk_size * dst_data_size,
                                          blk_size, 0, IC);
                });

                size_t tail_start = IS / blk_size * blk_size;
                reduce_kernel_process(in_ptr_n + tail_start * src_data_size, out_ptr_n + tail_start * dst_data_size,
                                      IS - tail_start, 0, IC);
            } else {
                for (size_t ic = 0; ic < IC; ic++) {
                    size_t oc = ReduceC ? 0 : ic; GET_PTR_NC_PLN;
                    for (size_t id = 0; id < ID; id++) {
                        size_t od = ReduceD ? 0 : id; GET_PTR_NCD_PLN;
                        for (size_t ih = 0; ih < IH; ih++) {
                            size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_PLN;
                            for (size_t ibw = 0; ibw < IW / blk_size; ibw++) {
                                size_t obw = ibw;
                                reduce_kernel_process(in_ptr_ncdh + ibw * blk_size * src_data_size,
                                                      out_ptr_ncdh + obw * blk_size * dst_data_size, blk_size, 0);
                            }
                            size_t tail_start = IW / blk_size * blk_size;
                            reduce_kernel_process(in_ptr_ncdh + tail_start * src_data_size, out_ptr_ncdh + tail_start * dst_data_size, IW - tail_start, 0);
                        }
                    }
                }
            }
        }
    }

    output_info_restore(&out_ptr);
    reduce_kernel_post_process(out_ptr);
}

void Reduce::reduce_BLK(const uint8_t *in_ptr, uint8_t *out_ptr) {
    size_t ICB = div_up(IC, blk_size);
    size_t OCB = div_up(OC, blk_size);
    output_info_reassign(out_ptr);
    init_dst_data(out_ptr, dst_size);

    for (size_t ib = 0; ib < IB; ib++) {
        size_t ob = ReduceN ? 0 : ib; GET_PTR_N_BLK;
        if (!ReduceC && !ReduceD && ReduceH && ReduceW) {
            if (!ReduceN || (ReduceN && ib == IB - 1)) {
                apply_division = getAlgorithm() == Algorithm::ReduceMean && attr.get()->post_ops_.len() == 0;
                apply_post_kernel = !apply_division;
            }
            parallel_for2d(ICB, ID, [&](size_t icb, size_t id) {
                size_t ocb = icb, od = id;
                GET_PTR_NCD_BASE_PTR_N_BLK;
                reduce_kernel_process(in_ptr_ncd, out_ptr_ncd, IH * IW * blk_size);
            });
        } else if (ReduceC && ReduceD && ReduceH && ReduceW) {
            if (ReduceAll_opt) {
                // reduce parallelly
                // step1: !ReduceC && ReduceD && ReduceH && ReduceW
                size_t prc_size = ICB * blk_size * dst_data_size;
                std::vector<uint8_t> vec_prc(prc_size);
                init_dst_data(vec_prc.data(), prc_size);
                uint8_t *out_ptr_n_cp = out_ptr_n;
                out_ptr_n = vec_prc.data();
                parallel_for(ICB, [&](size_t icb) {
                    size_t ocb = icb; GET_PTR_NC_BLK;
                    reduce_kernel_process(in_ptr_nc, out_ptr_nc, ID * IH * IW * blk_size);
                });
                // step2: ReduceC
                reduce_kernel_reassign();
                reduce_kernel_process(out_ptr_n, out_ptr_n_cp, ICB * blk_size);
                reduce_kernel_restore();
            } else {
                reduce_kernel_process(in_ptr_n, out_ptr_n, ICB * ID * IH * IW * blk_size);
            }
        } else if (ReduceW) {
            for (size_t icb = 0; icb < ICB; icb++) {
                size_t ocb = ReduceC ? 0 : icb; GET_PTR_NC_BLK;
                for (size_t id = 0; id < ID; id++) {
                    size_t od = ReduceD ? 0 : id; GET_PTR_NCD_BLK;
                    for (size_t ih = 0; ih < IH; ih++) {
                        size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_BLK;
                        reduce_kernel_process(in_ptr_ncdh, out_ptr_ncdh, IW * blk_size);
                    }
                }
            }
        } else if (ReduceC && !ReduceD && !ReduceH && !ReduceW) {
            reduce_stride = ID * IH * IW * blk_size;
            parallel_for3d(ID, IH, IW, [&](size_t id, size_t ih, size_t iw) {
                size_t icb = 0, ocb = 0; GET_PTR_NC_BLK;
                size_t od = id; GET_PTR_NCD_BLK;
                size_t oh = ih; GET_PTR_NCDH_BLK;
                size_t ow = iw; GET_PTR_NCDHW_BLK;
                reduce_kernel_process(in_ptr_ncdhw, out_ptr_ncdhw, blk_size, 0, ICB);
            });
        } else {
            for (size_t icb = 0; icb < ICB; icb++) {
                size_t ocb = ReduceC ? 0 : icb; GET_PTR_NC_BLK;
                for (size_t id = 0; id < ID; id++) {
                    size_t od = ReduceD ? 0 : id; GET_PTR_NCD_BLK;
                    for (size_t ih = 0; ih < IH; ih++) {
                        size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_BLK;
                        parallel_for(IW, [&](size_t iw) {
                            size_t ow = iw; GET_PTR_NCDHW_BLK;
                            reduce_kernel_process(in_ptr_ncdhw, out_ptr_ncdhw, blk_size);
                        });
                    }
                }
            }
        }
    }

    output_info_restore(&out_ptr);
    if (apply_post_kernel) {
        reduce_kernel_post_process(out_ptr);
    }
}

void Reduce::reduce_BLK_concern_padding(const uint8_t *in_ptr, uint8_t *out_ptr) {
    size_t ICB = div_up(IC, blk_size);
    size_t OCB = div_up(OC, blk_size);
    output_info_reassign(out_ptr);
    init_dst_data(out_ptr, dst_size);

    auto reduceSkipPadding = [&](const uint8_t *in_ptr_ncd, uint8_t *out_ptr_ncd, size_t ic) {
        size_t blk_valid_size = IC - ic;
        for (size_t ih = 0; ih < IH; ih++) {
            size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_BLK;
            for (size_t iw = 0; iw < IW; iw++) {
                size_t ow = ReduceW ? 0 : iw; GET_PTR_NCDHW_BLK;
                reduce_kernel_process(in_ptr_ncdhw, out_ptr_ncdhw, blk_valid_size);
            }
        }
    };

    for (size_t ib = 0; ib < IB; ib++) {
        size_t ob = ReduceN ? 0 : ib; GET_PTR_N_BLK;
        if (!ReduceD && ReduceH && ReduceW) {
            for (size_t icb = 0; icb < ICB; icb++) {
                size_t ocb = 0;;
                size_t ic = icb * blk_size;
                parallel_for(ID, [&](size_t id) {
                    size_t od = id; GET_PTR_NCD_BASE_PTR_N_BLK;
                    if (ic + blk_size <= IC) {
                        reduce_kernel_process(in_ptr_ncd, out_ptr_ncd, IH * IW * blk_size);
                    } else {
                        reduceSkipPadding(in_ptr_ncd, out_ptr_ncd, ic);
                    }
                });
            }
        } else if (ReduceD && ReduceH && ReduceW) {
            for (size_t icb = 0; icb < ICB; icb++) {
                size_t ocb = 0; GET_PTR_NC_BLK;
                size_t ic = icb * blk_size;
                if (ic + blk_size <= IC) {
                    reduce_kernel_process(in_ptr_nc, out_ptr_nc, ID * IH * IW * blk_size);
                } else {
                    for (size_t id = 0; id < ID; id++) {
                        size_t od = 0; GET_PTR_NCD_BLK;
                        reduceSkipPadding(in_ptr_ncd, out_ptr_ncd, ic);
                    }
                }
            }
        } else if (ReduceW) {
            for (size_t icb = 0; icb < ICB; icb++) {
                size_t ocb = 0; GET_PTR_NC_BLK;
                size_t ic = icb * blk_size;
                for (size_t id = 0; id < ID; id++) {
                    size_t od = ReduceD ? 0 : id; GET_PTR_NCD_BLK;
                    if (ic + blk_size <= IC) {
                        for (size_t ih = 0; ih < IH; ih++) {
                            size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_BLK;
                            reduce_kernel_process(in_ptr_ncdh, out_ptr_ncdh, IW * blk_size);
                        }
                    } else {
                        reduceSkipPadding(in_ptr_ncd, out_ptr_ncd, ic);
                    }
                }
            }
        } else {
            for (size_t icb = 0; icb < ICB; icb++) {
                size_t ocb = 0; GET_PTR_NC_BLK;
                size_t ic = icb * blk_size;
                for (size_t id = 0; id < ID; id++) {
                    size_t od = ReduceD ? 0 : id; GET_PTR_NCD_BLK;
                    if (ic + blk_size <= IC) {
                        for (size_t ih = 0; ih < IH; ih++) {
                            size_t oh = ReduceH ? 0 : ih; GET_PTR_NCDH_BLK;
                            parallel_for(IW, [&](size_t iw) {
                                size_t ow = iw; GET_PTR_NCDHW_BLK;
                                reduce_kernel_process(in_ptr_ncdhw, out_ptr_ncdhw, blk_size);
                            });
                        }
                    } else {
                        reduceSkipPadding(in_ptr_ncd, out_ptr_ncd, ic);
                    }
                }
            }
        }
    }

    output_info_restore(&out_ptr);
    reduce_kernel_post_process(out_ptr);
}

inline void Reduce::reduce_kernel_process(const uint8_t *in_p, uint8_t *out_p, size_t work_amount,
                                          size_t reduce_w, size_t work_batch, const int *tab_idx) {
    auto arg = JitReduceCallArgs();

    arg.src = static_cast<const void *>(in_p);
    arg.idx = tab_idx;
    arg.dst = static_cast<void *>(out_p);
    arg.work_amount = work_amount;
    arg.work_batch = work_batch;
    arg.reduce_w = reduce_w;
    arg.reduce_stride = reduce_stride;
    arg.can_divide = apply_division ? 1lu : 0lu;
    arg.divisor = in_out_divisor;

    (*reduce_kernel)(&arg);
}

inline void Reduce::reduce_kernel_post_process(uint8_t *out_ptr) {
    const uint8_t *in_ptr = fuse_low_precision ? static_cast<uint8_t *>(&intermediate_buf[0]) : nullptr;
    if (layout == ReduceLayoutType::reduce_ncsp) {
        const auto work_amount = OD * OH * OW;
        parallel_for2d(OB, OC, [&](size_t ob, size_t oc) {
            const uint8_t *in_p = in_ptr + (ob * OC + oc) * OD * OH * OW * intermediate_data_size;
            uint8_t *out_p = out_ptr + (ob * OC + oc) * work_amount * dst_data_size;
            auto arg = JitReducePostCallArgs();
            arg.src = static_cast<const void *>(in_p);
            arg.dst = static_cast<void *>(out_p);
            arg.oc_off = oc * sizeof(float);
            arg.channel_size = OC;
            arg.work_amount = work_amount;
            arg.divisor = in_out_divisor;
            arg.post_op_data = static_cast<const void **>(postOpsDataPtrs.data());

            (*reduce_post_kernel)(&arg);
        });
    } else if (layout == ReduceLayoutType::reduce_nspc) {
        size_t num_threads = static_cast<size_t>(parallel_get_max_threads());
        size_t OP = OB * OC >= num_threads ? OB * OC : OB * OC * OD;
        if (OP < num_threads && OW > blk_size)
            OP *= OH;
        const auto work_amount = OB * OC * OD * OH * OW / OP;
        parallel_for(OP, [&](size_t op) {
            const uint8_t *in_p = in_ptr + op * work_amount * intermediate_data_size;
            uint8_t *out_p = out_ptr + op * work_amount * dst_data_size;
            auto arg = JitReducePostCallArgs();

            arg.src = static_cast<const void *>(in_p);
            arg.dst = static_cast<void *>(out_p);
            arg.oc_off = 0;
            arg.channel_size = OW; // OW is related to nspc-ncsp dimension reinterpret
            arg.work_amount = work_amount;
            arg.divisor = in_out_divisor;
            arg.post_op_data = static_cast<const void **>(postOpsDataPtrs.data());

            (*reduce_post_kernel)(&arg);
        });
    } else {
        size_t OCB = div_up(OC, blk_size);
        const auto work_amount = OD * OH * OW * blk_size;
        parallel_for2d(OB, OCB, [&](size_t ob, size_t ocb) {
            const uint8_t *in_p = in_ptr + (ob * OCB + ocb) * OD * OH * OW * blk_size * intermediate_data_size;
            uint8_t *out_p = out_ptr + (ob * OCB + ocb) * work_amount * dst_data_size;
            auto arg = JitReducePostCallArgs();

            arg.src = static_cast<const void *>(in_p);
            arg.dst = static_cast<void *>(out_p);
            arg.reduce_c = ReduceC ? 1 : 0;
            arg.oc_off = ocb * blk_size * sizeof(float);
            arg.work_amount = work_amount;
            arg.divisor = in_out_divisor;
            arg.post_op_data = static_cast<const void **>(postOpsDataPtrs.data());

            (*reduce_post_kernel)(&arg);
        });
    }
}

inline void Reduce::reduce_kernel_reassign() {
    if (use_aux_kernel) {
        reduce_tmp_kernel = reduce_kernel;
        reduce_kernel = reduce_aux_kernel;
    }
}
inline void Reduce::reduce_kernel_restore() {
    if (use_aux_kernel) {
        reduce_kernel = reduce_tmp_kernel;
    }
}

inline void Reduce::output_info_reassign(uint8_t *out_ptr) {
    if (fuse_low_precision) {
        tmp_ptr = out_ptr;
        out_ptr = static_cast<uint8_t *>(&intermediate_buf[0]);
        tmp_prec = output_prec;
        output_prec = intermediate_prec;
        tmp_data_size = dst_data_size;
        dst_data_size = intermediate_data_size;
        tmp_size = dst_size;
        dst_size = intermediate_size;
    }
}
inline void Reduce::output_info_restore(uint8_t **out_ptr) {
    if (fuse_low_precision) {
        *out_ptr = tmp_ptr;
        output_prec = tmp_prec;
        dst_data_size = tmp_data_size;
        dst_size = tmp_size;
    }
}

template<typename T>
void Reduce::nspc2ncsp(uint8_t *proc_ptr, uint8_t *out_ptr) {
    // dimension reinterpret after nspc reusing routine reduce_PLN
    // demote -- nspc -- ncsp
    //  DIM0  --   B  --  B
    //  DIM1  --   C  --  W
    //  DIM2  --   D  --  C
    //  DIM3  --   H  --  D
    //  DIM4  --   W  --  H
    const size_t DIM0 = OB;
    const size_t DIM1 = OW;
    const size_t DIM2 = OC;
    const size_t DIM3 = OD;
    const size_t DIM4 = OH;
    const size_t stride1 = DIM2 * DIM3 * DIM4;
    const size_t stride0 = stride1 * DIM1;

    auto src_data = reinterpret_cast<const T *>(proc_ptr);
    auto dst_data = reinterpret_cast<T *>(out_ptr);
    parallel_for2d(DIM0, stride1, [&](size_t b, size_t j) {
        auto src_off = b * stride0 + j * DIM1;
        auto dst_off = b * stride0 + j;
        for (size_t dim1 = 0; dim1 < DIM1; dim1++) {
            dst_data[dst_off] = src_data[src_off];
            src_off++;
            dst_off += stride1;
        }
    });
}

template<typename T>
void Reduce::blocked2ncsp(uint8_t *proc_ptr, uint8_t *out_ptr) {
    const size_t DIM0 = OB;
    const size_t DIM1 = OC;
    const size_t DIM2 = OD;
    const size_t DIM3 = OH;
    const size_t DIM4 = OW;
    const size_t stride1 = DIM2 * DIM3 * DIM4;
    const size_t src_stride0 = stride1 * div_up(OC, blk_size) * blk_size;
    const size_t dst_stride0 = stride1 * DIM1;

    auto src_data = reinterpret_cast<const T *>(proc_ptr);
    auto dst_data = reinterpret_cast<T *>(out_ptr);
    parallel_for2d(DIM0, stride1, [&](size_t b, size_t j) {
        auto src_off = b * src_stride0 + j * blk_size;
        auto dst_off = b * dst_stride0 + j;
        for (size_t dim1 = 0; dim1 + blk_size <= DIM1; dim1 += blk_size) {
            for (size_t k = 0; k < blk_size; k++) {
                dst_data[dst_off] = src_data[src_off];
                src_off++;
                dst_off += stride1;
            }
            src_off += (stride1 - 1) * blk_size;
        }
        size_t tail = DIM1 % blk_size;
        for (size_t k = 0; k < tail; k++) {
            dst_data[dst_off] = src_data[src_off];
            src_off++;
            dst_off += stride1;
        }
    });
}

inline void Reduce::init_dst_data(uint8_t *out_ptr, size_t dst_size) {
    switch (algorithm) {
        case Algorithm::ReduceL1:
        case Algorithm::ReduceL2:
        case Algorithm::ReduceLogSum:
        case Algorithm::ReduceLogSumExp:
        case Algorithm::ReduceMean:
        case Algorithm::ReduceOr:
        case Algorithm::ReduceSum:
        case Algorithm::ReduceSumSquare:
            memset(out_ptr, 0, dst_size);
            break;
        case Algorithm::ReduceAnd:
        case Algorithm::ReduceProd:
            if (output_prec == Precision::FP64) {
                auto out_p = reinterpret_cast<double *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = static_cast<int64_t>(1); });
            } else if (output_prec == Precision::I64) {
                auto out_p = reinterpret_cast<int64_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = static_cast<int64_t>(1); });
            } else if (output_prec == Precision::FP32) {
                auto out_p = reinterpret_cast<float *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = static_cast<float>(1); });
            } else if (output_prec == Precision::I32) {
                auto out_p = reinterpret_cast<int32_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = static_cast<int32_t>(1); });
            } else if (output_prec == Precision::BF16) {
                auto out_p = reinterpret_cast<bfloat16_t*>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = static_cast<bfloat16_t>(1); });
            } else if (output_prec == Precision::U8) {
                auto out_p = reinterpret_cast<uint8_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = static_cast<uint8_t>(1); });
            } else if (output_prec == Precision::I8) {
                auto out_p = reinterpret_cast<int8_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = static_cast<int8_t>(1); });
            }
            break;
        case Algorithm::ReduceMax:
            if (output_prec == Precision::FP64) {
                auto out_p = reinterpret_cast<double *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<double>::lowest(); });
            } else if (output_prec == Precision::I64) {
                auto out_p = reinterpret_cast<int64_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<int64_t>::min(); });
            } else if (output_prec == Precision::FP32) {
                auto out_p = reinterpret_cast<float *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<float>::lowest(); });
            } else if (output_prec == Precision::I32) {
                auto out_p = reinterpret_cast<int32_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<int32_t>::min(); });
            } else if (output_prec == Precision::BF16) {
                auto out_p = reinterpret_cast<bfloat16_t*>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<bfloat16_t>::lowest(); });
            } else if (output_prec == Precision::U8) {
                auto out_p = reinterpret_cast<uint8_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<uint8_t>::min(); });
            } else if (output_prec == Precision::I8) {
                auto out_p = reinterpret_cast<int8_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<int8_t>::min(); });
            }
            break;
        case Algorithm::ReduceMin:
            if (output_prec == Precision::FP64) {
                auto out_p = reinterpret_cast<double *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<double>::max(); });
            } else if (output_prec == Precision::I64) {
                auto out_p = reinterpret_cast<int64_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<int64_t>::max(); });
            } else if (output_prec == Precision::FP32) {
                auto out_p = reinterpret_cast<float *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<float>::max(); });
            } else if (output_prec == Precision::I32) {
                auto out_p = reinterpret_cast<int32_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<int32_t>::max(); });
            } else if (output_prec == Precision::BF16) {
                auto out_p = reinterpret_cast<bfloat16_t*>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<bfloat16_t>::max(); });
            } else if (output_prec == Precision::U8) {
                auto out_p = reinterpret_cast<uint8_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<uint8_t>::max(); });
            } else if (output_prec == Precision::I8) {
                auto out_p = reinterpret_cast<int8_t *>(out_ptr);
                parallel_for(dst_size / dst_data_size, [&](size_t i) { out_p[i] = std::numeric_limits<int8_t>::max(); });
            }
            break;
        default:
            THROW_CPU_NODE_ERR << " gets unsupported reduce mode.";
    }
}

inline void Reduce::create_hybrid_working_memory() {
    auto rank = getInputShapeAtPort(REDUCE_DATA).getRank();
    dnnl::memory::format_tag format =
            (layout == ReduceLayoutType::reduce_nspc) ? (rank == 4 ? dnnl::memory::format_tag::nhwc : dnnl::memory::format_tag::ndhwc)
                    : (rank == 4 ? (x64::mayiuse(x64::avx512_core) ? dnnl::memory::format_tag::nChw16c : dnnl::memory::format_tag::nChw8c)
                                 : (x64::mayiuse(x64::avx512_core) ? dnnl::memory::format_tag::nCdhw16c : dnnl::memory::format_tag::nCdhw8c));
    auto prc_dims = rank == 4 ? std::vector<size_t>{OB, OC, OH, OW} : std::vector<size_t>{OB, OC, OD, OH, OW};
    auto desc = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(prc_dims), DnnlExtensionUtils::IEPrecisionToDataType(output_prec), format);
    prc_mem = dnnl::memory(desc, getEngine());
    dst_size = desc.get_size();
}

inline void Reduce::create_opt_working_memory() {
    if (fuse_low_precision) {
        intermediate_size = dst_size * sizeof(float) / dst_data_size;
        if (intermediate_size > intermediate_buf.size()) {
            intermediate_buf.resize(intermediate_size);
        }
    }

    ReduceDH_opt = layout == ReduceLayoutType::reduce_nspc && support_split &&
                   !ReduceC && ReduceD && ReduceH && !ReduceW && IC == 1 && ID > 1;
    if (ReduceDH_opt) {
        PD = ID;
        PW = IW / blk_size * blk_size;
        prc_data_size = intermediate_data_size;
        prc_size = PD * PW * prc_data_size;
        if (prc_size > vec_reduceDH_prc.size()) {
            vec_reduceDH_prc.resize(prc_size);
        }
        return;
    }

    ReduceCDW_opt = layout == ReduceLayoutType::reduce_ncsp && support_split &&
                    ReduceC && ReduceD && !ReduceH && ReduceW;
    if (ReduceCDW_opt) {
        PH = IH;
        PW = IW;
        prc_data_size = intermediate_data_size;
        prc_size = PH * PW * prc_data_size;
        if (prc_size > vec_reduceCDW_prc.size()) {
            vec_reduceCDW_prc.resize(prc_size);
        }
    }
}

inline void Reduce::calc_process_dst_dims(std::vector<int64_t> &reduce_axes, const SizeVector &dst_dims) {
    std::set<size_t> axes;
    SizeVector out_dims;
    process_dst_dims.clear();
    axes_for_reduction.clear();
    for (auto &axis : reduce_axes) {
        if (axis < 0)
            axis += src_dims.size();
        if (static_cast<size_t>(axis) > src_dims.size())
            THROW_CPU_NODE_ERR << " exceeds data tensor dimension on index to reduce";
        axes.insert(static_cast<size_t>(axis));
    }
    for (size_t i = 0; i < src_dims.size(); i++) {
        bool found = false;
        for (auto axis : axes) {
            if (i == axis) {
                found = true;
                break;
            }
        }
        if (found) {
            if (keep_dims) out_dims.push_back(1);
            process_dst_dims.push_back(1);
            axes_for_reduction.push_back(i);
        } else {
            out_dims.push_back(src_dims[i]);
            process_dst_dims.push_back(src_dims[i]);
        }
    }
    if (jit_mode && jit_beyond_5D) {
        if (std::accumulate(out_dims.begin(), out_dims.end(), size_t(1), std::multiplies<size_t>()) !=
            std::accumulate(dst_dims.begin(), dst_dims.end(), size_t(1), std::multiplies<size_t>()))
            THROW_CPU_NODE_ERR << "gets incorrect number of output dimensions!";
    } else {
        for (size_t i = 0; i < std::min(out_dims.size(), dst_dims.size()); i++) {
            if (out_dims[i] != dst_dims[i])
                THROW_CPU_NODE_ERR << "gets incorrect number of output dimensions!";
        }
    }
}

inline void Reduce::set_reduce_dim_flags() {
    size_t dims_size = src_dims.size();
    if (dims_size == 5) {
        SET_SRC_DIM_VALUE(src_dims[0], src_dims[1], src_dims[2], src_dims[3], src_dims[4]);
        SET_DST_DIM_VALUE(process_dst_dims[0], process_dst_dims[1], process_dst_dims[2], process_dst_dims[3], process_dst_dims[4]);
    } else if (dims_size == 4) {
        SET_SRC_DIM_VALUE(src_dims[0], src_dims[1], 1, src_dims[2], src_dims[3]);
        SET_DST_DIM_VALUE(process_dst_dims[0], process_dst_dims[1], 1, process_dst_dims[2], process_dst_dims[3]);
    } else if (dims_size == 3) {
        SET_SRC_DIM_VALUE(1, src_dims[0], 1, src_dims[1], src_dims[2]);
        SET_DST_DIM_VALUE(1, process_dst_dims[0], 1, process_dst_dims[1], process_dst_dims[2]);
    } else if (dims_size == 2) {
        SET_SRC_DIM_VALUE(1, 1, 1, src_dims[0], src_dims[1]);
        SET_DST_DIM_VALUE(1, 1, 1, process_dst_dims[0], process_dst_dims[1]);
    } else {
        SET_SRC_DIM_VALUE(1, src_dims[0], 1, 1, 1);
        SET_DST_DIM_VALUE(1, process_dst_dims[0], 1, 1, 1);
    }

    // must be done before the following dimension change
    if (is_hybrid_layout) {
        create_hybrid_working_memory();
    }

    // Reducing a dimesion in nspc layout can be treated as reducing another dimension in ncsp layout,
    // eg. reducing C in nspc can be treated as reducing W in ncsp layout, so that the routine reduce_PLN can be reused.
    // nspc -- ncsp
    //    D -- C
    //    H -- D
    //    W -- H
    //    C -- W
    if (layout == ReduceLayoutType::reduce_nspc) {
        size_t ITmp = IC; IC = ID; ID = IH; IH = IW; IW = ITmp;
        size_t OTmp = OC; OC = OD; OD = OH; OH = OW; OW = OTmp;
    }

    ReduceN = IB != OB && OB == 1;
    ReduceC = IC != OC && OC == 1;
    ReduceD = ID != OD && OD == 1;
    ReduceH = IH != OH && OH == 1;
    ReduceW = IW != OW && OW == 1;

    // suit for parallel
    if (ReduceH && IW == 1) {
        ReduceW = true;
    }
    if (ReduceC && ReduceH && ID == 1) {
        ReduceD = true;
    }

    // must be done after the above dimension change
    create_opt_working_memory();

    ReduceAll_opt = layout == ReduceLayoutType::reduce_blocked && support_split &&
                    ReduceC && ReduceD && ReduceH && ReduceW;
    if (!set_use_aux_kernel) {
        use_aux_kernel = (ReduceDH_opt || ReduceCDW_opt || ReduceAll_opt) && precision_change;
        set_use_aux_kernel = true;
    }
}

inline void Reduce::reduce_ref(const float *in_ptr, float *out_ptr) {
    switch (algorithm) {
        case Algorithm::ReduceAnd:
            reduce_ref_process(in_ptr, out_ptr, 1, [](float x, float y)->float { return x && y; });
            break;
        case Algorithm::ReduceL1:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float old, float y)->float { return old + (y >= 0 ? y : -y); });
            break;
        case Algorithm::ReduceL2:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float old, float y)->float { return old + y * y; });
            break;
        case Algorithm::ReduceLogSum:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float x, float y)->float { return x + y; });
            break;
        case Algorithm::ReduceLogSumExp:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float old, float y)->float { return old + expf(y); });
            break;
        case Algorithm::ReduceMax:
            reduce_ref_process(in_ptr, out_ptr, std::numeric_limits<float>::lowest(),
                                                    [](float x, float y)->float { return x > y ? x : y; });
            break;
        case Algorithm::ReduceMean:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float x, float y)->float { return x + y; });
            break;
        case Algorithm::ReduceMin:
            reduce_ref_process(in_ptr, out_ptr, std::numeric_limits<float>::max(),
                                                    [](float x, float y)->float { return x < y ? x : y; });
            break;
        case Algorithm::ReduceOr:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float x, float y)->float { return x || y; });
            break;
        case Algorithm::ReduceProd:
            reduce_ref_process(in_ptr, out_ptr, 1, [](float x, float y)->float { return x * y; });
            break;
        case Algorithm::ReduceSum:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float x, float y)->float { return x + y; });
            break;
        case Algorithm::ReduceSumSquare:
            reduce_ref_process(in_ptr, out_ptr, 0, [](float old, float y)->float { return old + y * y; });
            break;
        default:
            THROW_CPU_NODE_ERR << "gets unsupported reduce mode.";
    }
}

void Reduce::reduce_ref_process(const float *in_ptr, float *out_ptr, float init_value, std::function<float(float, float)> func) {
    size_t work_amount_dst = 1, reduced_dims_work_amount = 1;
    for (size_t i = 0; i < process_dst_dims.size(); i++)
        work_amount_dst *= process_dst_dims[i];
    for (size_t i = 0; i < src_dims.size(); i++)
        reduced_dims_work_amount *= src_dims[i];
    reduced_dims_work_amount /= work_amount_dst;

    SizeVector src_strides = getParentEdgeAt(REDUCE_DATA)->getMemory().getDescWithType<BlockedMemoryDesc>()->getStrides();
    parallel_nt(0, [&](const int ithr, const int nthr) {
        int j;
        size_t i, start = 0, end = 0;
        SizeVector dst_counters(process_dst_dims.size(), 0);
        splitter(work_amount_dst, nthr, ithr, start, end);
        for (j = process_dst_dims.size() - 1, i = start; j >= 0; j--) {
            dst_counters[j] = i % process_dst_dims[j];
            i /= process_dst_dims[j];
        }
        for (size_t src_idx = 0, dst_idx = start; dst_idx < end; ++dst_idx) {
            float reduce_prod = init_value;
            bool update_idx = true;
            SizeVector src_counters = dst_counters;
            for (i = 0; i < reduced_dims_work_amount; ++i) {
                if (update_idx) {
                    src_idx = 0;
                    for (j = 0; j < static_cast<int>(src_dims.size()); ++j)
                        src_idx += (src_counters[j] % src_dims[j]) * src_strides[j];
                    update_idx = false;
                }
                reduce_prod = func(reduce_prod, in_ptr[src_idx]);
                for (j = axes_for_reduction.size() - 1; j >= 0; j--) {
                    src_counters[axes_for_reduction[j]]++;
                    if (src_counters[axes_for_reduction[j]] < src_dims[axes_for_reduction[j]]) {
                        src_idx += src_strides[axes_for_reduction[j]];
                        break;
                    } else {
                        src_counters[axes_for_reduction[j]] = 0;
                        update_idx = true;
                    }
                }
            }
            out_ptr[dst_idx] = reduce_prod;
            for (j = process_dst_dims.size() - 1; j >= 0; j--) {
                dst_counters[j]++;
                if (dst_counters[j] < process_dst_dims[j])
                    break;
                else
                    dst_counters[j] = 0;
            }
        }
    });

    reduce_ref_map(out_ptr, work_amount_dst, reduced_dims_work_amount);
}

inline void Reduce::reduce_ref_map(float *out_ptr, size_t work_amount_dst, size_t reduced_dims_work_amount) {
    switch (algorithm) {
        case Algorithm::ReduceAnd:
        case Algorithm::ReduceL1:
        case Algorithm::ReduceMax:
        case Algorithm::ReduceMin:
        case Algorithm::ReduceOr:
        case Algorithm::ReduceProd:
        case Algorithm::ReduceSum:
        case Algorithm::ReduceSumSquare:
            break;
        case Algorithm::ReduceL2:
            parallel_for(work_amount_dst, [&](size_t i) {
                out_ptr[i] = std::sqrt(out_ptr[i]);
            });
            break;
        case Algorithm::ReduceLogSum:
        case Algorithm::ReduceLogSumExp:
            parallel_for(work_amount_dst, [&](size_t i) {
                out_ptr[i] = logf(out_ptr[i]);
            });
            break;
        case Algorithm::ReduceMean:
            parallel_for(work_amount_dst, [&](size_t i) {
                out_ptr[i] /= reduced_dims_work_amount;
            });
            break;
        default:
            THROW_CPU_NODE_ERR << "gets unsupported reduce mode.";
    }
}

void Reduce::setPostOps(dnnl::primitive_attr &attr, const VectorDims &postOpDims, bool initWeights) {
    dnnl::post_ops ops;
    postOpsDataPtrs.clear();
    for (auto &node : fusedWith) {
        auto* fakeQuantizeNode = dynamic_cast<FakeQuantize *>(node.get());
        if (fakeQuantizeNode) {
            fakeQuantizeNode->appendPostOps(ops, {}, postOpsDataPtrs);
            continue;
        }

        auto* eltwiseNode = dynamic_cast<Eltwise *>(node.get());
        if (eltwiseNode) {
            eltwiseNode->appendPostOps(ops, postOpDims, postOpsDataPtrs, getFusingAxis());
            continue;
        }
        IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType()) << " node is not implemented";
    }

    attr.set_post_ops(ops);
}

void Reduce::setJITBeyond5D() {
    jit_beyond_5D = false;
    if (getInputShapeAtPort(REDUCE_DATA).getRank() > 5) {
        for (auto &axis : raw_axes) {
            if (axis < 0)
                axis += static_cast<int>(getInputShapeAtPort(REDUCE_DATA).getRank());
        }

        if (raw_axes.size() <= 1) {
            jit_beyond_5D = true;
        } else {
            for (size_t i = 1; i < raw_axes.size(); i++) {
                if (raw_axes[i] != raw_axes[i - 1] + 1) {
                    jit_beyond_5D = false;
                    break;
                }
                jit_beyond_5D = true;
            }
        }
    }
}

std::vector<int64_t> Reduce::update_src_dims() {
    std::vector<int64_t> reduce_axes = raw_axes;

    if (reduce_axes.size() < 1)
        return reduce_axes;

    size_t axis_dim = 1;
    size_t outer_dim = 1;
    size_t inner_dim = 1;
    int outer_end = reduce_axes[0];
    int inner_start = reduce_axes[reduce_axes.size() - 1];
    for (size_t i = 0; i < src_dims.size(); i++) {
        if (static_cast<int>(i) < outer_end) {
            outer_dim *= src_dims[i];
        } else if (static_cast<int>(i) > inner_start) {
            inner_dim *= src_dims[i];
        } else {
            axis_dim *= src_dims[i];
        }
    }

    reduce_axes.clear();
    reduce_axes.push_back(1);

    src_dims.clear();
    src_dims.push_back(outer_dim);
    src_dims.push_back(axis_dim);
    src_dims.push_back(inner_dim);

    return reduce_axes;
}

bool Reduce::canApplyJIT(const Precision &input_prec, const Precision &output_prec) const {
    static const Precision supportedPrecisions[] = {
            Precision::I64,
            Precision::FP32,
            Precision::BF16,
            Precision::I32,
            Precision::I8,
            Precision::U8
    };

    return (x64::mayiuse(x64::sse41)) && (getInputShapeAtPort(REDUCE_DATA).getRank() <= 5 || jit_beyond_5D) &&
           std::find(std::begin(supportedPrecisions), std::end(supportedPrecisions), input_prec) != std::end(supportedPrecisions) &&
           std::find(std::begin(supportedPrecisions), std::end(supportedPrecisions), output_prec) != std::end(supportedPrecisions);
}

int Reduce::getFusingAxis() const {
    int channelAxis = 1;
    if (!keep_dims) {
        for (auto &raw_axis : raw_axes) {
            int axis = raw_axis >= 0 ? raw_axis : raw_axis + static_cast<int>(getInputShapeAtPort(REDUCE_DATA).getRank());
            if (axis == 1) {
                // channel axis has been reduced and doesn't exist any more
                channelAxis = -1;
                break;
            } else if (axis == 0) {
                channelAxis = 0;
            }
        }
    }
    return channelAxis;
}

bool Reduce::canFuse(const NodePtr& node) const {
    const auto& input_prec = getOriginalInputPrecisionAtPort(REDUCE_DATA);
    const auto& output_prec = getOriginalOutputPrecisionAtPort(0);
    if (!canApplyJIT(input_prec, output_prec) || jit_beyond_5D || algorithm == Algorithm::ReduceAnd || algorithm == Algorithm::ReduceOr) {
        return false;
    }

    if (one_of(8, input_prec.size(), output_prec.size())) {
        return false;
    }

    return canFuseSimpleOperation(node);
}

bool Reduce::created() const {
    return getType() == Type::Reduce;
}
