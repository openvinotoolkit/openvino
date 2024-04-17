// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "subgraph.h"

#include "common/primitive_hashing_utils.hpp"
#include "dnnl_extension_utils.h"
#include "emitters/snippets/x64/cpu_generator.hpp"
#include "onednn/dnnl.h"
#include "openvino/core/parallel.hpp"
#include "openvino/core/rt_info.hpp"
#include "shape_inference/custom/subgraph.hpp"
#include "snippets/utils.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/pass/hash.hpp"
#include "snippets/pass/matmul_to_brgemm.hpp"
#include "snippets/pass/propagate_precision.hpp"
#include "snippets/pass/positioned_pass.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/optimize_domain.hpp"
#include "snippets/lowered/pass/insert_loops.hpp"
#include "snippets/lowered/pass/mark_loops.hpp"
#include "transformations/defs.hpp"
#include "transformations/cpu_opset/common/pass/convert_to_swish_cpu.hpp"
#include "transformations/snippets/x64/pass/lowered/brgemm_blocking.hpp"
#include "transformations/snippets/x64/pass/lowered/fuse_load_store_and_convert.hpp"
#include "transformations/snippets/x64/pass/lowered/set_brgemm_copy_b_buffers_shape.hpp"
#include "transformations/snippets/x64/pass/mul_add_to_fma.hpp"
#include "transformations/snippets/x64/pass/remove_converts.hpp"
#include "transformations/snippets/x64/pass/set_brgemm_cpu_blocking_params.hpp"
#include "transformations/snippets/x64/pass/brgemm_to_brgemm_cpu.hpp"
#include "transformations/snippets/x64/pass/enforce_precision.hpp"
#include "transformations/snippets/x64/shape_inference.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/ngraph_utils.hpp"

#include <algorithm>
#include <array>
#include <vector>

#if defined(__linux__) && defined(SNIPPETS_DEBUG_CAPS)
#include "emitters/snippets/x64/jit_segfault_detector_emitter.hpp"
#include <signal.h>
std::mutex err_print_lock;
#endif

#ifdef SNIPPETS_LIBXSMM_TPP
#include "transformations/tpp/x64/pass/brgemm_to_brgemm_tpp.hpp"
#include "transformations/tpp/x64/pass/eltwise_to_eltwise_tpp.hpp"
#include "transformations/tpp/x64/pass/scalar_to_scalar_tpp.hpp"
#include "transformations/tpp/x64/pass/lowered/set_tpp_leading_dim.hpp"
#endif

using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov {
namespace intel_cpu {
namespace node {
namespace {

struct SubgraphKey {
    Subgraph::SubgraphAttrs attrs;

    size_t hash() const;
    bool operator==(const SubgraphKey& rhs) const;
};

size_t SubgraphKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    for (const auto& blockedDim : attrs.inMemBlockedDims)
        seed = get_vector_hash(seed, blockedDim);
    for (const auto& order : attrs.inMemOrders)
        seed = get_vector_hash(seed, order);
    for (const auto& prec : attrs.inMemPrecs)
        seed = hash_combine(seed, prec.hash());

    for (const auto& blockedDim : attrs.outMemBlockedDims)
        seed = get_vector_hash(seed, blockedDim);
    for (const auto& order : attrs.outMemOrders)
        seed = get_vector_hash(seed, order);
    for (const auto& prec : attrs.outMemPrecs)
        seed = hash_combine(seed, prec.hash());

    seed = hash_combine(seed, attrs.bodyHash);

    return seed;
}

bool SubgraphKey::operator==(const SubgraphKey& rhs) const {
    if (attrs.bodyHash != rhs.attrs.bodyHash)
        return false;
    if (attrs.inMemBlockedDims.size() != rhs.attrs.inMemBlockedDims.size() ||
        attrs.inMemOrders.size() != rhs.attrs.inMemOrders.size() ||
        attrs.inMemPrecs.size() != rhs.attrs.inMemPrecs.size())
        return false;
    if (attrs.outMemBlockedDims.size() != rhs.attrs.outMemBlockedDims.size() ||
        attrs.outMemOrders.size() != rhs.attrs.outMemOrders.size() ||
        attrs.outMemPrecs.size() != rhs.attrs.outMemPrecs.size())
        return false;

    for (size_t i = 0; i < attrs.inMemBlockedDims.size(); i++) {
        if (!(attrs.inMemBlockedDims[i] == rhs.attrs.inMemBlockedDims[i]))
            return false;
    }
    for (size_t i = 0; i < attrs.outMemBlockedDims.size(); i++) {
        if (!(attrs.outMemBlockedDims[i] == rhs.attrs.outMemBlockedDims[i]))
            return false;
    }
    for (size_t i = 0; i < attrs.inMemOrders.size(); i++) {
        if (!(attrs.inMemOrders[i] == rhs.attrs.inMemOrders[i]))
            return false;
    }
    for (size_t i = 0; i < attrs.outMemOrders.size(); i++) {
        if (!(attrs.outMemOrders[i] == rhs.attrs.outMemOrders[i]))
            return false;
    }
    for (size_t i = 0; i < attrs.inMemPrecs.size(); i++) {
        if (!(attrs.inMemPrecs[i] == rhs.attrs.inMemPrecs[i]))
            return false;
    }
    for (size_t i = 0; i < attrs.outMemPrecs.size(); i++) {
        if (!(attrs.outMemPrecs[i] == rhs.attrs.outMemPrecs[i]))
            return false;
    }

    return true;
}
} // namespace

Subgraph::Subgraph(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
        : Node(op, context, SnippetShapeInferFactory(op)) {
    host_isa = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ?
        dnnl::impl::cpu::x64::avx512_core : dnnl::impl::cpu::x64::avx2;
    const auto& tmp_snippet = ov::as_type_ptr<snippets::op::Subgraph>(op);
    OPENVINO_ASSERT(tmp_snippet, "Attempt to create Subgraph node from an invalid op type");
    snippetAttrs.snippet = tmp_snippet->clone();
    snippetAttrs.bodyHash = get_body_hash(tmp_snippet);

#if defined(OPENVINO_ARCH_X86_64)
    snippetAttrs.snippet->set_generator(std::make_shared<CPUGenerator>(host_isa));
#else
    OPENVINO_THROW("CPU plugin: Snippets code-generator is not supported on non-x64 platforms");

#endif // OPENVINO_ARCH_X86_64

    // Note: we have to update shapeInfer, so it uses the per-thread op::Subgraph copy
    shapeInference = SnippetShapeInferFactory(snippetAttrs.snippet).makeShapeInfer();
    is_dynamic = isDynamicNgraphNode(op);
}

uint64_t Subgraph::get_body_hash(const std::shared_ptr<snippets::op::Subgraph>& snippet) {
    uint64_t seed = 0;
    ov::snippets::pass::Hash hash_function(seed);
    hash_function.run_on_model(snippet->body_ptr());
    return seed;
}

void Subgraph::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const std::set<ov::element::Type> supportedPrecisions =
        {ov::element::f32, ov::element::i32, ov::element::bf16, ov::element::f16, ov::element::i8, ov::element::u8};

    bool dimRanksAreEqual = true;
    for (size_t i = 0; dimRanksAreEqual && i < inputShapes.size(); i++) {
        for (size_t j = 0; dimRanksAreEqual && j < outputShapes.size(); j++) {
            if (inputShapes[i].getRank() != outputShapes[j].getRank())
                dimRanksAreEqual = false;
        }
    }

    const size_t ndims = outputShapes[0].getRank();
    // Domain sensitive operations support only Planar layout
    const bool isOnlyPlanarApplicable = snippetAttrs.snippet->has_domain_sensitive_ops();
    const bool isChannelsFirstApplicable = dnnl::impl::utils::one_of(ndims, 1u, 2u, 3u, 4u, 5u) && dimRanksAreEqual && !isOnlyPlanarApplicable;
    // Todo: Snippets currently don't support per-channel broadcasting of Blocked descriptors because
    //  canonicalization can't distinguish between <N, C, H, W, c> and <N, C, D, H, W> cases.
    //  See snippets::op::Subgraph::canonicalize for details.
    bool isBlockedApplicable = dnnl::impl::utils::one_of(ndims,  3u, 4u, 5u) && dimRanksAreEqual && !isOnlyPlanarApplicable;

    for (const auto& inShape : inputShapes) {
        if (isDynamic && inShape.getRank() != 1)
            isBlockedApplicable = isBlockedApplicable && inShape.getMinDims()[1] != Shape::UNDEFINED_DIM && inShape.getMinDims()[1] > 1;
    }

    enum LayoutType {
        Planar,
        ChannelsFirst,
        Blocked
    };
    auto initDesc = [&] (LayoutType lt) -> NodeDesc {
        auto createMemoryDesc = [lt](const Shape &shape, ov::element::Type prc, size_t offset) -> std::shared_ptr<CpuBlockedMemoryDesc> {
            const auto &dims = shape.getDims();
            if (lt == ChannelsFirst && shape.getRank() != 1) {
                auto ndims = shape.getRank();
                VectorDims order(ndims);
                std::iota(order.begin(), order.end(), 0);
                if (ndims > 1) {
                    order.erase(order.begin() + 1);
                    order.push_back(1);
                }

                VectorDims blocks(ndims);
                for (size_t i = 0; i < order.size(); i++) {
                    blocks[i] = dims[order[i]];
                }

                return std::make_shared<CpuBlockedMemoryDesc>(prc, shape, blocks, order, offset);
            } else if (lt == Blocked && shape.getRank() != 1 && (shape.getMinDims()[1] != Shape::UNDEFINED_DIM && shape.getMinDims()[1] > 1)) {
                size_t blockSize = mayiuse(dnnl::impl::cpu::x64::avx512_core) ? 16 : 8;

                VectorDims blocks = dims;
                VectorDims order(blocks.size());
                std::iota(order.begin(), order.end(), 0);

                blocks[1] = dims[1] != Shape::UNDEFINED_DIM ? div_up(blocks[1], blockSize) : Shape::UNDEFINED_DIM;
                blocks.push_back(blockSize);
                order.push_back(1);

                return std::make_shared<CpuBlockedMemoryDesc>(prc, shape, blocks, order, offset);
            } else {
                VectorDims blocks = dims;
                VectorDims order(blocks.size());
                std::iota(order.begin(), order.end(), 0);

                return std::make_shared<CpuBlockedMemoryDesc>(prc, shape, blocks, order, offset);
            }
        };

        size_t offset = 0;
        NodeConfig config;
        config.inConfs.resize(inputShapes.size());
        for (size_t i = 0; i < inputShapes.size(); i++) {
            const auto originalInputPrecision = getOriginalInputPrecisionAtPort(i);
            const auto precision = ((originalInputPrecision == ov::element::f32) &&
                                     context->getConfig().inferencePrecision == ov::element::bf16 &&
                                     snippetAttrs.snippet->has_domain_sensitive_ops()) ?
                static_cast<ov::element::Type>(ov::element::bf16) :
                originalInputPrecision;
            if (supportedPrecisions.count(precision) == 0)
                OPENVINO_THROW("Subgraph node with name `", getName(), "` doesn't support ", precision, " precision.");

            const auto equalPrecisions = getOriginalOutputPrecisions().size() == 1 &&
                    precision == getOriginalOutputPrecisionAtPort(0);

            BlockedMemoryDesc::CmpMask inputMask = BlockedMemoryDesc::SKIP_OFFSET_MASK;
            PortConfig portConfig;
            portConfig.inPlace((!i && canBeInPlace() && equalPrecisions) ? 0 : -1);
            portConfig.constant(false);
            if (inputShapes[i].getDims()[0] == 1) {
                inputMask.reset(0); // accepts any stride on batch axis
            }
            portConfig.setMemDesc(createMemoryDesc(inputShapes[i], precision, offset), inputMask);
            config.inConfs[i] = portConfig;
        }
        config.outConfs.resize(outputShapes.size());
        for (size_t i = 0; i < outputShapes.size(); i++) {
            auto precision = getOriginalOutputPrecisionAtPort(i);
            if (supportedPrecisions.count(precision) == 0)
                OPENVINO_THROW("Subgraph node with name `", getName(), "` doesn't support ", precision, " precision.");

            BlockedMemoryDesc::CmpMask outputMask = BlockedMemoryDesc::SKIP_OFFSET_MASK;
            PortConfig portConfig;
            portConfig.inPlace(-1);
            portConfig.constant(false);
            if (outputShapes[i].getDims()[0] == 1) {
                outputMask.reset(0); // accepts any stride on batch axis
            }
            portConfig.setMemDesc(createMemoryDesc(outputShapes[i], precision, offset), outputMask);
            config.outConfs[i] = portConfig;
        }

        impl_desc_type impl_type = impl_desc_type::unknown;
        if (mayiuse(x64::avx512_core)) {
            impl_type = impl_desc_type::jit_avx512;
        } else if (mayiuse(x64::avx2)) {
            impl_type = impl_desc_type::jit_avx2;
        }
        return {config, impl_type};
    };

    if (isChannelsFirstApplicable)
        supportedPrimitiveDescriptors.emplace_back(initDesc(ChannelsFirst));
    if (isBlockedApplicable)
        supportedPrimitiveDescriptors.emplace_back(initDesc(Blocked));
    supportedPrimitiveDescriptors.emplace_back(initDesc(Planar));
}

void Subgraph::selectOptimalPrimitiveDescriptor() {
    selectPreferPrimitiveDescriptor(getImplPriority(), true);
}

void Subgraph::initOptimalPrimitiveDescriptor() {
    const auto isPlanar = [](const VectorDims& order ) {
        for (size_t i = 0; i < order.size(); ++i)
            if (order[i] != i)
                return false;
        return true;
    };
    Node::initOptimalPrimitiveDescriptor();
    // memory order and precision is determined now, there is no need to prepare for each dynamic shapes.
    const auto config = getSelectedPrimitiveDescriptor()->getConfig();
    inputNum = config.inConfs.size();
    snippets::op::Subgraph::BlockedShapeVector in_blocked_shapes;
    snippetAttrs.inMemPrecs.resize(inputNum);
    snippetAttrs.inMemOrders.resize(inputNum);
    in_blocked_shapes.reserve(inputNum);
    snippetAttrs.has_non_planar_inputs = false;
    for (size_t i = 0; i < inputNum; i++) {
        const auto& memDesc = config.inConfs[i].getMemDesc();
        snippetAttrs.inMemPrecs[i] = memDesc->getPrecision();
        const auto& blockedDesc = memDesc->as<BlockedMemoryDesc>();
        const auto& order = blockedDesc->getOrder();
        snippetAttrs.inMemOrders[i] = order;
        snippetAttrs.has_non_planar_inputs |= !isPlanar(order);
        in_blocked_shapes.emplace_back(blockedDesc->getBlockDims(), order);
    }
    outputNum = config.outConfs.size();
    snippetAttrs.outMemPrecs.resize(outputNum);
    snippetAttrs.outMemOrders.resize(outputNum);
    for (size_t i = 0; i < outputNum; i++) {
        snippetAttrs.outMemPrecs[i] = config.outConfs[i].getMemDesc()->getPrecision();
        snippetAttrs.outMemOrders[i] = config.outConfs[i].getMemDesc()->as<BlockedMemoryDesc>()->getOrder();
    }
    // reserve fixed size.
    snippetAttrs.inMemBlockedDims.resize(inputNum);
    snippetAttrs.outMemBlockedDims.resize(outputNum);
    srcMemPtrs.resize(inputNum);
    dstMemPtrs.resize(outputNum);

    // here we should perform all shape-agnostic snippets passes
    // * canonicalization (RankNormalization insert)
    // * precision propagation & align element types
    // * data flow optimizations
    // The result of these transformations will be reused by all shapes
    std::vector<ov::snippets::pass::Manager::PositionedPassBase> backend_passes;
#if defined(OPENVINO_ARCH_X86_64)
    using PassPosition = ov::snippets::pass::PassPosition;
    using Place = PassPosition::Place;
#   define SNIPPETS_REGISTER_PASS_ABSOLUTE(PASS_PLACE, PASS, ...) \
            backend_passes.emplace_back(PassPosition(PASS_PLACE), std::make_shared<PASS>(__VA_ARGS__))
#   define SNIPPETS_REGISTER_PASS_RELATIVE(PASS_PLACE, TARGET_PASS, PASS, ...) \
            backend_passes.emplace_back(PassPosition(PASS_PLACE, TARGET_PASS::get_type_info_static()), std::make_shared<PASS>(__VA_ARGS__))
#else
#    define SNIPPETS_REGISTER_PASS_ABSOLUTE(PASS_PLACE, PASS, ...)
#    define SNIPPETS_REGISTER_PASS_RELATIVE(PASS_PLACE, TARGET_PASS, PASS, ...)
#endif  // OPENVINO_ARCH_X86_64

    SNIPPETS_REGISTER_PASS_ABSOLUTE(Place::PipelineStart, ConvertToSwishCPU);
    if (context->getConfig().inferencePrecision == ov::element::bf16 && snippetAttrs.snippet->has_domain_sensitive_ops()) {
        // enforce BF16 precisions to supported operations
        // MatMul has to be decomposed to Brgemm operations before enforcement
        // Note, MatMul decomposition will be run later again for case if BF16 enforcement is not happened
        SNIPPETS_REGISTER_PASS_ABSOLUTE(Place::PipelineStart, ov::snippets::pass::MatMulToBrgemm);
        SNIPPETS_REGISTER_PASS_RELATIVE(Place::After, ov::snippets::pass::MatMulToBrgemm,
                                        pass::EnforcePrecision, element::f32, element::bf16);
    }

    SNIPPETS_REGISTER_PASS_RELATIVE(Place::Before, ov::snippets::pass::PropagatePrecision,
                                    ov::intel_cpu::pass::BrgemmToBrgemmCPU);
    SNIPPETS_REGISTER_PASS_RELATIVE(Place::After, ov::intel_cpu::pass::BrgemmToBrgemmCPU,
                                    ov::intel_cpu::pass::SetBrgemmCPUBlockingParams);
    SNIPPETS_REGISTER_PASS_ABSOLUTE(Place::PipelineEnd, ov::intel_cpu::pass::RemoveConverts);
    SNIPPETS_REGISTER_PASS_ABSOLUTE(Place::PipelineEnd, ov::intel_cpu::pass::MulAddToFMA);

#ifdef SNIPPETS_LIBXSMM_TPP
    SNIPPETS_REGISTER_PASS_RELATIVE(Place::Before, ov::intel_cpu::pass::BrgemmToBrgemmCPU,
                                    ov::intel_cpu::tpp::pass::BrgemmToBrgemmTPP);
    // Note: There could be several ConvertConstantsToScalars instances in the pipeline
    SNIPPETS_REGISTER_PASS_ABSOLUTE(Place::PipelineEnd, ov::intel_cpu::tpp::pass::ScalarToScalarTPP);
    SNIPPETS_REGISTER_PASS_RELATIVE(Place::After, ov::intel_cpu::tpp::pass::BrgemmToBrgemmTPP,
                                    ov::intel_cpu::tpp::pass::EltwiseToEltwiseTPP);
#endif

#undef SNIPPETS_REGISTER_PASS

    std::vector<ov::element::Type> input_precisions;
    std::vector<ov::element::Type> output_precisions;
    input_precisions.reserve(inputNum);
    for (const auto& p :  snippetAttrs.inMemPrecs) {
        input_precisions.push_back(p);
    }
    output_precisions.reserve(outputNum);
    for (const auto& p :  snippetAttrs.outMemPrecs)
        output_precisions.push_back(p);

    snippetAttrs.snippet->data_flow_transformations(in_blocked_shapes, input_precisions, output_precisions, backend_passes);
    // Note: minimal JIT work amount is a predefined value that describes the number of kernel iterations (work amount)
    // needed to cover kernel call overhead. It is used for balancing between parallel and JIT work amounts in domain optimization.
#ifdef SNIPPETS_LIBXSMM_TPP
    const auto& lir = snippetAttrs.snippet->convert_body_to_linear_ir(static_cast<size_t>(parallel_get_max_threads()), 256,
                                                                      std::make_shared<snippets::CPUShapeInferSnippetsFactory>());
    lir->set_loop_depth(std::min(2ul, lir->get_master_shape().size()));
#else
    snippetAttrs.snippet->convert_body_to_linear_ir(static_cast<size_t>(parallel_get_max_threads()), 256,
                                                    std::make_shared<snippets::CPUShapeInferSnippetsFactory>());
#endif
}

ov::element::Type Subgraph::getRuntimePrecision() const {
    std::vector<ov::element::Type> inputPrecisions;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge && parentEdge->getStatus() == Edge::Status::Validated && !parentEdge->getParent()->isConstant()) {
            inputPrecisions.emplace_back(DnnlExtensionUtils::DataTypeToElementType((parentEdge->getMemoryPtr()->getDataType())));
        }
    }

    return getMaxPrecision(inputPrecisions);
}

void Subgraph::prepareParams() {
    for (size_t i = 0; i < inputNum; i++)
        snippetAttrs.inMemBlockedDims[i] = getParentEdgeAt(i)->getMemory().getDescWithType<BlockedMemoryDesc>()->getBlockDims();
    for (size_t i = 0; i < outputNum; i++)
        snippetAttrs.outMemBlockedDims[i] = getChildEdgeAt(i)->getMemory().getDescWithType<BlockedMemoryDesc>()->getBlockDims();

    SubgraphKey key = {snippetAttrs};

    auto builder = [this](const SubgraphKey& key) -> std::shared_ptr<SubgraphExecutor> {
        std::shared_ptr<SubgraphExecutor> executor =
                std::make_shared<SubgraphJitExecutor>(key.attrs, is_dynamic);
        return executor;
    };

    auto getOrCreateExecutor = [this, &key, &builder]() {
        auto cache = context->getParamsCache();
        auto result = cache->getOrCreate(key, builder);
        execPtr = result.first;
        if (!execPtr) {
            OPENVINO_THROW("Executor is not created for node ", getName(), ".");
        }
    };

#ifndef SNIPPETS_DEBUG_CAPS
    getOrCreateExecutor();
#else
    snippets::lowered::Config config;
    if (config.perf_count_mode == snippets::lowered::PerfCountMode::Disabled) {
        getOrCreateExecutor();
    } else {
        // in case perf count is enabled, disable executor cache by default to not mix up perf counters for different subgraphs.
        execPtr = std::make_shared<SubgraphJitExecutor>(key.attrs, is_dynamic);
    }
#endif
}

bool Subgraph::needPrepareParams() const {
    auto jit_executor = dynamic_cast<SubgraphJitExecutor*>(execPtr.get());
    return inputShapesModified() || (jit_executor && !jit_executor->schedule_created());
}

bool Subgraph::canBeInPlace() const {
    if (isDynamic || getParentEdgeAt(0)->getParent()->getType() == Type::Input) {
        return false;
    }
    if (getChildEdges().size() != 1) {
        return false;
    }

    for (auto& parentEdge : getParentEdges()) {
        auto parent = parentEdge.lock()->getParent();
        if (parent->getChildEdges().size() != 1)
            return false;

        // WA to prevent memory corruption caused by inplace feature
        if (parent->getType() == Type::Concatenation) {
            for (auto& parentParentEdge : parent->getParentEdges()) {
                auto parentParent = parentParentEdge.lock()->getParent();
                if (parentParent->getChildEdges().size() != 1)
                    return false;
            }
        }
    }
    return getInputShapeAtPort(0) == getOutputShapeAtPort(0);
}

bool Subgraph::created() const {
    return getType() == Type::Subgraph;
}

void Subgraph::execute(dnnl::stream strm) {
    if (!execPtr) {
        OPENVINO_THROW("Can't execute Subgraph node. Primitive didn't created");
    }
    for (size_t i = 0; i < inputNum; i++)
        srcMemPtrs[i] = getSrcMemoryAtPort(i);
    for (size_t i = 0; i < outputNum; i++)
        dstMemPtrs[i] = getDstMemoryAtPort(i);

    execPtr->exec(srcMemPtrs, dstMemPtrs);
}

void Subgraph::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void Subgraph::SubgraphJitExecutor::exec(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) {
    if (schedule.lowering_result.compiled_snippet->empty()) {
        OPENVINO_THROW("Subgraph can't use Optimized implementation and can't fallback to reference");
    }
    auto initStartMemoryOffsets = [this, &inMemPtrs, &outMemPtrs]() {
        for (size_t i = 0; i < numInput; i++) {
            start_offset_in[i] =
                    static_cast<ptrdiff_t>(inMemPtrs[i]->getDescWithType<BlockedMemoryDesc>()->getOffsetPadding() * dataSize[i]);
        }
        for (size_t i = 0; i < numOutput; i++) {
            start_offset_out[i] =
                    static_cast<ptrdiff_t>(outMemPtrs[i]->getDescWithType<BlockedMemoryDesc>()->getOffsetPadding() * dataSize[i + numInput]);
        }
    };
    // initialize start offsets to src and dst memory
    // Needs to be done for every set of infer, as data memory ptrs could've updated
    initStartMemoryOffsets();

    if (tensorRank == rank6D) {
        schedule_6d(inMemPtrs, outMemPtrs);
    } else {
        schedule_nt(inMemPtrs, outMemPtrs);
    }
}

void Subgraph::SubgraphJitExecutor::update_ptrs(jit_snippets_call_args& call_args,
    const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) {
    for (size_t i = 0; i < inMemPtrs.size(); i++)
        call_args.src_ptrs[i] = inMemPtrs[i]->getDataAs<const uint8_t>() + start_offset_in[i];

    for (size_t i = 0; i < outMemPtrs.size(); i++)
        call_args.dst_ptrs[i] = outMemPtrs[i]->getDataAs<uint8_t>() + start_offset_out[i];

    if (buffer_scratchpad_size > 0) {
        call_args.buffer_scratchpad_ptr =
                reinterpret_cast<uint8_t*>(buffer_scratchpad.data()) + parallel_get_thread_num() * buffer_scratchpad_size;
    }
}

#if defined(__linux__) && defined(SNIPPETS_DEBUG_CAPS)
void Subgraph::SubgraphJitExecutor::segfault_detector() {
    const auto target = std::dynamic_pointer_cast<const CPUTargetMachine>(snippetAttrs.snippet->get_generator()->get_target_machine());
    if (target && target->debug_config.enable_segfault_detector) {
        __sighandler_t signal_handler = [](int signal) {
            std::lock_guard<std::mutex> guard(err_print_lock);
            if (auto segfault_detector_emitter = ov::intel_cpu::g_custom_segfault_handler->local())
                std::cout << segfault_detector_emitter->info() << std::endl;
            auto tid = parallel_get_thread_num();
            OPENVINO_THROW("Segfault was caught by the signal handler in subgraph node execution on thread " + std::to_string(tid));
        };
        struct sigaction new_handler{};
        new_handler.sa_handler = signal_handler;
        sigaction(SIGSEGV, &new_handler, nullptr);
    }
}
#endif

void Subgraph::SubgraphJitExecutor::schedule_6d(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) {
    const auto& dom = parallel_exec_domain;
    // < N, C, H, W > < 1, 1, N, C*H*W>
    const auto& callable = schedule.get_callable<kernel>();
#if defined(__linux__) && defined(SNIPPETS_DEBUG_CAPS)
    segfault_detector();
#endif
    parallel_for5d(dom[0], dom[1], dom[2], dom[3], dom[4],
        [&](int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4) {
            int64_t indexes[] = {d0, d1, d2, d3, d4};
            jit_snippets_call_args call_args;
            update_ptrs(call_args, inMemPtrs, outMemPtrs);
            callable(&call_args, indexes);
        });
}

void Subgraph::SubgraphJitExecutor::schedule_nt(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) {
    const auto& work_size = parallel_exec_domain;
#if defined(__linux__) && defined(SNIPPETS_DEBUG_CAPS)
    segfault_detector();
#endif
    parallel_nt(0, [&](const int ithr, const int nthr) {
        jit_snippets_call_args call_args;
        update_ptrs(call_args, inMemPtrs, outMemPtrs);

        size_t start = 0, end = 0;
        splitter(harnessWorkAmount, nthr, ithr, start, end);

        std::vector<int64_t> indexes(work_size.size() - 1, 0);
        for (size_t iwork = start; iwork < end; ++iwork) {
            size_t tmp = iwork;
            for (ptrdiff_t j = static_cast<ptrdiff_t>(work_size.size()) - 2; j >= 0; j--) {
                indexes[j] = static_cast<int64_t>(tmp % work_size[j]);
                tmp /= work_size[j];
            }

            schedule.get_callable<kernel>()(&call_args, indexes.data());
        }
    });
}

Subgraph::SubgraphExecutor::SubgraphExecutor(SubgraphAttrs attrs, bool is_dynamic)
    : snippetAttrs(std::move(attrs)), is_dynamic(is_dynamic) {}

Subgraph::SubgraphJitExecutor::SubgraphJitExecutor(SubgraphAttrs attrs, bool is_dynamic) :
    SubgraphExecutor(std::move(attrs), is_dynamic) {
    numInput = snippetAttrs.inMemBlockedDims.size();
    numOutput = snippetAttrs.outMemBlockedDims.size();
    start_offset_in.resize(numInput);
    start_offset_out.resize(numOutput);

    // todo: snippets don't support backend-provided blocking, so we need to reshape body
    //  using blocked shapes first. This can be removed after [121670]
    if (snippetAttrs.has_non_planar_inputs) {
        std::vector<snippets::VectorDimsRef> in_shapes;
        for (const auto& s : snippetAttrs.inMemBlockedDims)
            in_shapes.emplace_back(s);
        snippetAttrs.snippet->shape_infer(in_shapes);
    }
    const VectorDims& canonicalShape = snippetAttrs.snippet->infer_master_shape();

    // initialize by maximum output dimension. Dimensions of outputs should be broadcastable
    tensorRank = std::max(static_cast<size_t>(rank6D), canonicalShape.size());
    auto initDataSizes = [this]() {
        dataSize.resize(numInput + numOutput);
        for (size_t i = 0; i < numInput; i++)
            dataSize[i] = snippetAttrs.inMemPrecs[i].size();
        for (size_t i = 0; i < numOutput; i++)
            dataSize[i + numInput] = snippetAttrs.outMemPrecs[i].size();
    };
    initDataSizes();

    if (snippets::utils::is_dynamic_vdims(canonicalShape))
        OPENVINO_THROW("Snippets: Canonicalization returned dynamic shape in static pipeline");

    // generate
    jit_snippets_compile_args jcp;
    jcp.parallel_executor_ndims = tensorRank;
    generate(&jcp);
    buffer_scratchpad_size = schedule.lowering_result.buffer_scratchpad_size;
    buffer_scratchpad.resize(buffer_scratchpad_size * parallel_get_max_threads(), 0);
    parallel_exec_domain = schedule.parallel_exec_domain;
    harnessWorkAmount = std::accumulate(parallel_exec_domain.begin(), parallel_exec_domain.end(), 1, std::multiplies<size_t>());
    parallel_exec_domain = getNormalizedDimsBySize(parallel_exec_domain, tensorRank);
}

void Subgraph::SubgraphJitExecutor::generate(const jit_snippets_compile_args* jcp) {
    std::vector<ov::snippets::lowered::pass::PassPipeline::PositionedPassLowered> backend_passes;

#if defined(OPENVINO_ARCH_X86_64)
    using PassPosition = ov::snippets::pass::PassPosition;
    using Place = PassPosition::Place;
#   define SNIPPETS_REGISTER_PASS_RELATIVE(PASS_PLACE, TARGET_PASS, PASS, ...) \
            backend_passes.emplace_back(PassPosition(PASS_PLACE, TARGET_PASS::get_type_info_static()), std::make_shared<PASS>(__VA_ARGS__))
#else
#    define SNIPPETS_REGISTER_PASS_RELATIVE(PASS_PLACE, TARGET_PASS, PASS, ...)
#endif  // OPENVINO_ARCH_X86_64

    SNIPPETS_REGISTER_PASS_RELATIVE(Place::After, ov::snippets::lowered::pass::MarkLoops,
                                    ov::intel_cpu::pass::BrgemmBlocking);
    SNIPPETS_REGISTER_PASS_RELATIVE(Place::After, ov::snippets::lowered::pass::InsertLoops,
                                    ov::intel_cpu::pass::FuseLoadStoreConvert);
    SNIPPETS_REGISTER_PASS_RELATIVE(Place::After, ov::intel_cpu::pass::FuseLoadStoreConvert,
                                    ov::intel_cpu::pass::SetBrgemmCopyBBuffersShape);

    auto lowering_config = std::make_shared<ov::snippets::lowered::pass::PassConfig>();
#ifdef SNIPPETS_LIBXSMM_TPP
    // Note: temporary disabled. Re-enable after ticket 132833 is resolved
    lowering_config->disable<ov::snippets::lowered::pass::OptimizeDomain>();
    SNIPPETS_REGISTER_PASS_RELATIVE(Place::After, ov::intel_cpu::pass::FuseLoadStoreConvert,
                                    ov::intel_cpu::tpp::pass::SetTPPLeadingDim);
#endif

    schedule = snippetAttrs.snippet->generate_from_linear_ir(lowering_config,
                                                             backend_passes,
                                                             reinterpret_cast<const void*>(jcp));

#undef SNIPPETS_REGISTER_PASS_RELATIVE
}

bool Subgraph::SubgraphJitExecutor::schedule_created() {
    return !schedule.lowering_result.compiled_snippet->empty();
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
