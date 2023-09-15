// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph.h"

#include <ie_parallel.hpp>

#include <vector>
#include <algorithm>
#include <array>

#include <onednn/dnnl.h>
#include <dnnl_extension_utils.h>

#include <ngraph/rt_info.hpp>
#include <ie_ngraph_utils.hpp>

#include <snippets/op/subgraph.hpp>
#include <snippets/lowered/pass/optimize_domain.hpp>
#include "snippets/pass/matmul_to_brgemm.hpp"
#include "utils/cpu_utils.hpp"
#include "emitters/x64/cpu_generator.hpp"
#include "transformations/snippets/x64/pass/lowered/set_brgemm_copy_b_buffers_shape.hpp"
#include "transformations/snippets/x64/pass/lowered/fuse_load_store_and_convert.hpp"
#include "transformations/snippets/x64/pass/lowered/brgemm_blocking.hpp"
#include "transformations/snippets/x64/pass/mul_add_to_fma.hpp"
#include "transformations/snippets/x64/pass/brgemm_to_brgemm_cpu.hpp"
#include "transformations/snippets/x64/pass/remove_converts.hpp"
#include "transformations/snippets/x64/pass/enforce_precision.hpp"
#include "transformations/snippets/x64/pass/set_brgemm_cpu_blocking_params.hpp"
#include "transformations/snippets/x64/shape_inference.hpp"
#include "transformations/cpu_opset/common/pass/convert_to_swish_cpu.hpp"
#include "transformations/defs.hpp"
#include "shape_inference/custom/subgraph.hpp"
#include <common/primitive_hashing_utils.hpp>
#include "snippets/pass/hash.hpp"

#include <signal.h>

using namespace InferenceEngine;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov {
namespace intel_cpu {
namespace node {
namespace {

struct SnippetKey {
    Snippet::SnippetAttrs attrs;

    size_t hash() const;
    bool operator==(const SnippetKey& rhs) const;
};

size_t SnippetKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    for (const auto& blockedDim : attrs.inMemBlockedDims)
        seed = get_vector_hash(seed, blockedDim);
    for (const auto& order : attrs.inMemOrders)
        seed = get_vector_hash(seed, order);
    for (const auto& prec : attrs.inMemPrecs)
        seed = hash_combine(seed, prec.getPrecVal());

    for (const auto& blockedDim : attrs.outMemBlockedDims)
        seed = get_vector_hash(seed, blockedDim);
    for (const auto& order : attrs.outMemOrders)
        seed = get_vector_hash(seed, order);
    for (const auto& prec : attrs.outMemPrecs)
        seed = hash_combine(seed, prec.getPrecVal());

    seed = hash_combine(seed, attrs.bodyHash);

    return seed;
}

bool SnippetKey::operator==(const SnippetKey& rhs) const {
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

Snippet::Snippet(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
        : Node(op, context, SnippetShapeInferFactory(op)) {
    host_isa = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ?
        dnnl::impl::cpu::x64::avx512_core : dnnl::impl::cpu::x64::avx2;
    const auto& tmp_snippet = ov::as_type_ptr<snippets::op::Subgraph>(op);
    OPENVINO_ASSERT(tmp_snippet, "Attempt to create Snippet node from an invalid op type");
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

uint64_t Snippet::get_body_hash(const std::shared_ptr<snippets::op::Subgraph>& snippet) {
    uint64_t seed = 0;
    ov::snippets::pass::Hash hash_function(seed);
    hash_function.run_on_model(snippet->body_ptr());
    return seed;
}

void Snippet::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const std::set<Precision> supportedPrecisions = { Precision::FP32, Precision::I32, Precision::BF16, Precision::FP16, Precision::I8, Precision::U8 };

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
        auto createMemoryDesc = [lt](const Shape &shape, Precision prc, size_t offset) -> std::shared_ptr<CpuBlockedMemoryDesc> {
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
            const auto precision = ((originalInputPrecision == InferenceEngine::Precision::FP32) &&
                                     context->getConfig().inferencePrecision == ov::element::bf16 &&
                                     snippetAttrs.snippet->has_domain_sensitive_ops()) ?
                static_cast<InferenceEngine::Precision>(InferenceEngine::Precision::BF16) :
                originalInputPrecision;
            if (supportedPrecisions.count(precision) == 0)
                IE_THROW() << "Subgraph node with name `" << getName() << "` doesn't support " << precision << " precision.";

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
                IE_THROW() << "Subgraph node with name `" << getName() << "` doesn't support " << precision << " precision.";

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

void Snippet::selectOptimalPrimitiveDescriptor() {
    selectPreferPrimitiveDescriptor(getImplPriority(), true);
}

void Snippet::initOptimalPrimitiveDescriptor() {
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
    using Manager = snippets::pass::Manager;
    std::vector<Manager::PositionedPass> backend_passes;
#if defined(OPENVINO_ARCH_X86_64)
    using PassPosition = snippets::pass::Manager::PassPosition;
    using Place = snippets::pass::Manager::PassPosition::Place;
#   define SNIPPETS_REGISTER_PASS(PASS_POS, PASS, ...) \
            backend_passes.emplace_back(PASS_POS, std::make_shared<PASS>(__VA_ARGS__))
#else
#    define SNIPPETS_REGISTER_PASS(PASS_POS, PASS, ...)
#endif  // OPENVINO_ARCH_X86_64

    SNIPPETS_REGISTER_PASS(PassPosition(Place::PipelineStart), ConvertToSwishCPU);
    if (context->getConfig().inferencePrecision == ov::element::bf16 && snippetAttrs.snippet->has_domain_sensitive_ops()) {
        // enforce BF16 precisions to supported operations
        // MatMul has to be decomposed to Brgemm operations before enforcement
        // Note, MatMul decomposition will be run later again for case if BF16 enforcement is not happened
        SNIPPETS_REGISTER_PASS(PassPosition(Place::PipelineStart), ov::snippets::pass::MatMulToBrgemm);
        SNIPPETS_REGISTER_PASS(PassPosition(Place::After, "MatMulToBrgemm"), pass::EnforcePrecision, element::f32, element::bf16);
    }

    SNIPPETS_REGISTER_PASS(PassPosition(Place::Before, "PropagatePrecision"), ov::intel_cpu::pass::BrgemmToBrgemmCPU);
    SNIPPETS_REGISTER_PASS(PassPosition(Place::Before, "PropagatePrecision"), ov::intel_cpu::pass::SetBrgemmCPUBlockingParams);

    SNIPPETS_REGISTER_PASS(PassPosition(Place::PipelineEnd), ov::intel_cpu::pass::RemoveConverts);
    SNIPPETS_REGISTER_PASS(PassPosition(Place::PipelineEnd), ov::intel_cpu::pass::MulAddToFMA);

#undef SNIPPETS_REGISTER_PASS

    std::vector<ov::element::Type> input_precisions;
    std::vector<ov::element::Type> output_precisions;
    input_precisions.reserve(inputNum);
    for (const auto& p :  snippetAttrs.inMemPrecs) {
        input_precisions.push_back(InferenceEngine::details::convertPrecision(p));
    }
    output_precisions.reserve(outputNum);
    for (const auto& p :  snippetAttrs.outMemPrecs)
        output_precisions.push_back(InferenceEngine::details::convertPrecision(p));

    snippetAttrs.snippet->data_flow_transformations(in_blocked_shapes, input_precisions, output_precisions, backend_passes);
    snippetAttrs.snippet->convert_body_to_linear_ir(std::make_shared<snippets::CPUShapeInferSnippetsFactory>());
}

InferenceEngine::Precision Snippet::getRuntimePrecision() const {
    std::vector<InferenceEngine::Precision> inputPrecisions;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge && parentEdge->getStatus() == Edge::Status::Validated && !parentEdge->getParent()->isConstant()) {
            inputPrecisions.emplace_back(DnnlExtensionUtils::DataTypeToIEPrecision((parentEdge->getMemoryPtr()->getDataType())));
        }
    }

    return getMaxPrecision(inputPrecisions);
}

void Snippet::prepareParams() {
    for (size_t i = 0; i < inputNum; i++)
        snippetAttrs.inMemBlockedDims[i] = getParentEdgesAtPort(i)[0]->getMemory().getDescWithType<BlockedMemoryDesc>()->getBlockDims();
    for (size_t i = 0; i < outputNum; i++)
        snippetAttrs.outMemBlockedDims[i] = getChildEdgesAtPort(i)[0]->getMemory().getDescWithType<BlockedMemoryDesc>()->getBlockDims();

    SnippetKey key = {snippetAttrs};

    auto builder = [this](const SnippetKey& key) -> std::shared_ptr<SnippetExecutor> {
        std::shared_ptr<SnippetExecutor> executor =
                std::make_shared<SnippetJitExecutor>(key.attrs, is_dynamic, context->getConfig().inferencePrecision == ov::element::bf16);
        return executor;
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);
    execPtr = result.first;
    if (!execPtr) {
        IE_THROW() << "Executor is not created for node " << getName() << ".";
    }
}

bool Snippet::needPrepareParams() const {
    auto jit_executor = dynamic_cast<SnippetJitExecutor*>(execPtr.get());
    return inputShapesModified() || (jit_executor && !jit_executor->schedule_created());
}

bool Snippet::canBeInPlace() const {
    if (isDynamic || getParentEdgesAtPort(0)[0]->getParent()->getType() == Type::Input) {
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

bool Snippet::created() const {
    return getType() == Type::Subgraph;
}

void Snippet::execute(dnnl::stream strm) {
    if (!execPtr) {
        IE_THROW() << "Can't execute Subgraph node. Primitive didn't created";
    }
    for (size_t i = 0; i < inputNum; i++)
        srcMemPtrs[i] = getParentEdgeAt(i)->getMemoryPtr();
    for (size_t i = 0; i < outputNum; i++)
        dstMemPtrs[i] = getChildEdgeAt(i)->getMemoryPtr();

    execPtr->exec(srcMemPtrs, dstMemPtrs);
}

void Snippet::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void Snippet::SnippetJitExecutor::exec(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) {
    if (schedule.lowering_result.compiled_snippet->empty()) {
        IE_THROW() << "Snippet can't use Optimized implementation and can't fallback to reference";
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

void Snippet::SnippetJitExecutor::update_ptrs(jit_snippets_call_args& call_args,
    const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) {
    for (size_t i = 0; i < inMemPtrs.size(); i++)
        call_args.src_ptrs[i] = reinterpret_cast<const uint8_t*>(inMemPtrs[i]->getData()) + start_offset_in[i];

    for (size_t i = 0; i < outMemPtrs.size(); i++)
        call_args.dst_ptrs[i] = reinterpret_cast<uint8_t*>(outMemPtrs[i]->getData()) + start_offset_out[i];

    if (buffer_scratchpad_size > 0) {
        call_args.buffer_scratchpad_ptr =
                reinterpret_cast<uint8_t*>(buffer_scratchpad.data()) + parallel_get_thread_num() * buffer_scratchpad_size;
    }
}

void Snippet::SnippetJitExecutor::schedule_6d(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) {
    const auto& dom = parallel_exec_domain;
    // < N, C, H, W > < 1, 1, N, C*H*W>
    const auto& callable = schedule.get_callable<kernel>();
    parallel_for5d(1, 1, 1, 1, 1,
        [&](int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4) {
            int64_t indexes[] = {d0, d1, d2, d3, d4};
            jit_snippets_call_args call_args;
            update_ptrs(call_args, inMemPtrs, outMemPtrs);
#ifndef _WIN32
            
            __sighandler_t signal_handler = [](int signal) {
                std::cerr << "Segfault was caught by the signal handler.\n";
                ov::intel_cpu::g_debug_err_handler->print_debug_info();
            };
            struct sigaction new_handler{};
            new_handler.sa_handler = signal_handler;
            sigaction(SIGSEGV, &new_handler, nullptr);
#endif
            callable(indexes, &call_args);
        });
}

void Snippet::SnippetJitExecutor::schedule_nt(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) {
    const auto& work_size = parallel_exec_domain;
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
#ifndef _WIN32
            __sighandler_t signal_handler = [](int signal) {
                std::cerr << "Segfault was caught by the signal handler.\n";
                ov::intel_cpu::g_debug_err_handler->print_debug_info();
            };
            struct sigaction new_handler{};
            new_handler.sa_handler = signal_handler;
            sigaction(SIGSEGV, &new_handler, nullptr);
#endif
            schedule.get_callable<kernel>()(indexes.data(), &call_args);
        }
    });
}

Snippet::SnippetExecutor::SnippetExecutor(SnippetAttrs attrs, bool is_dynamic, bool enforceBF16)
    : snippetAttrs(std::move(attrs)), is_dynamic(is_dynamic), enforceBF16(enforceBF16) {}

Snippet::SnippetJitExecutor::SnippetJitExecutor(SnippetAttrs attrs, bool is_dynamic, bool enforceBF16) :
    SnippetExecutor(std::move(attrs), is_dynamic, enforceBF16) {
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

    if (std::any_of(canonicalShape.begin(), canonicalShape.end(),
                    [](size_t x){return x == snippets::IShapeInferSnippets::DYNAMIC_DIMENSION;}))
        IE_THROW() << "Snippets: Canonicalization returned dynamic shape in static pipeline";
    snippetAttrs.snippet->set_min_parallel_work_amount(static_cast<size_t>(parallel_get_max_threads()));
    // Note: minimal JIT work amount is a predefined value that describes the number of kernel iterations (work amount)
    // needed to cover kernel call overhead. It is used for balancing between parallel and JIT work amounts in domain optimization.
    snippetAttrs.snippet->set_min_jit_work_amount(256);

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

void Snippet::SnippetJitExecutor::generate(const jit_snippets_compile_args* jcp) {
    ov::snippets::lowered::pass::PassPipeline control_flow_markup_pipeline;
    CPU_REGISTER_PASS_X64(control_flow_markup_pipeline, ov::intel_cpu::pass::BrgemmBlocking)

    ov::snippets::lowered::pass::PassPipeline control_flow_pipeline;
    CPU_REGISTER_PASS_X64(control_flow_pipeline, ov::intel_cpu::pass::FuseLoadStoreConvert)
    CPU_REGISTER_PASS_X64(control_flow_pipeline, ov::intel_cpu::pass::SetBrgemmCopyBBuffersShape);
    schedule = snippetAttrs.snippet->generate_from_linear_ir(control_flow_markup_pipeline,
                                                             control_flow_pipeline,
                                                             reinterpret_cast<const void*>(jcp));
}

bool Snippet::SnippetJitExecutor::schedule_created() {
    return !schedule.lowering_result.compiled_snippet->empty();
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
