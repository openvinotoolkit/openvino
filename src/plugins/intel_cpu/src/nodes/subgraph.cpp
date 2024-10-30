// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "subgraph.h"

#include "common/primitive_hashing_utils.hpp"
#include "dnnl_extension_utils.h"
#include "onednn/dnnl.h"
#include "openvino/core/parallel.hpp"
#include "openvino/core/rt_info.hpp"
#include "shape_inference/custom/subgraph.hpp"
#include "snippets/utils/utils.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/pass/hash.hpp"
#include "snippets/pass/matmul_to_brgemm.hpp"
#include "snippets/pass/propagate_precision.hpp"
#include "snippets/pass/positioned_pass.hpp"
#include "snippets/pass/canonicalization.hpp"
#include "snippets/pass/analyze_broadcastable_inputs.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/optimize_domain.hpp"
#include "snippets/lowered/pass/insert_loops.hpp"
#include "snippets/lowered/pass/mark_loops.hpp"
#include "snippets/lowered/pass/insert_buffers.hpp"
#include "transformations/defs.hpp"
#include "transformations/cpu_opset/common/pass/convert_to_swish_cpu.hpp"
#include "transformations/snippets/common/pass/mul_add_to_fma.hpp"

#if defined(OPENVINO_ARCH_ARM64)
#include "emitters/snippets/aarch64/cpu_generator.hpp"
#include "transformations/snippets/aarch64/shape_inference.hpp"
#else
#include "emitters/snippets/x64/cpu_generator.hpp"
#include "transformations/snippets/x64/pass/lowered/brgemm_cpu_blocking.hpp"
#include "transformations/snippets/x64/pass/lowered/fuse_load_store_and_convert.hpp"
#include "transformations/snippets/x64/pass/lowered/insert_brgemm_copy_b_buffers.hpp"
#include "transformations/snippets/x64/pass/remove_converts.hpp"
#include "transformations/snippets/x64/pass/brgemm_to_brgemm_cpu.hpp"
#include "transformations/snippets/x64/pass/enforce_precision.hpp"
#include "transformations/snippets/x64/shape_inference.hpp"
#endif

#include "utils/cpu_utils.hpp"
#include "utils/ngraph_utils.hpp"

#include <algorithm>
#include <array>
#include <vector>

#if defined(__linux__) && defined(OPENVINO_ARCH_X86_64) && defined(SNIPPETS_DEBUG_CAPS)
#include "emitters/snippets/x64/jit_segfault_detector_emitter.hpp"
#include <signal.h>
std::mutex err_print_lock;
#endif

#ifdef SNIPPETS_LIBXSMM_TPP
#include "transformations/tpp/x64/pass/brgemm_to_brgemm_tpp.hpp"
#include "transformations/tpp/x64/pass/eltwise_to_eltwise_tpp.hpp"
#include "transformations/tpp/x64/pass/scalar_to_scalar_tpp.hpp"
#include "transformations/tpp/x64/pass/lowered/set_tpp_leading_dim.hpp"
#include "transformations/tpp/x64/pass/lowered/brgemm_tpp_blocking.hpp"
#include "transformations/tpp/x64/pass/fuse_tpp_to_equations.hpp"
#endif

namespace ov {
namespace intel_cpu {
namespace node {
namespace {

// Class for Subgraphs with static shapes
class SubgraphStaticExecutor : public Subgraph::SubgraphExecutor {
public:
    SubgraphStaticExecutor(const std::shared_ptr<Subgraph::SubgraphAttrs>& snippet_attrs,
                           const std::shared_ptr<Subgraph::SubgraphCodeGenerator>& snippet,
                           const std::vector<ptrdiff_t>& start_offset_in,
                           const std::vector<ptrdiff_t>& start_offset_out,
                           const std::shared_ptr<CPURuntimeConfig>& snippet_config,
                           const BufferScratchpadAllocator& allocator)
    : SubgraphExecutor(snippet_attrs, snippet, start_offset_in, start_offset_out, snippet_config, allocator) {}

    void exec(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override  {
        const auto& callable = m_schedule->get_callable<kernel>();

        auto initializer = [&](jit_snippets_call_args& call_args, size_t ithr) {
            init_call_args(call_args, inMemPtrs, outMemPtrs, ithr);
        };
        auto caller = [&](jit_snippets_call_args& call_args, const size_t* indexes) {
            callable(&call_args, indexes);
        };

        if (m_parallel_exec_domain.size() == rank6D) {
            parallel_for6d(initializer, caller);
        } else {
            parallel_forNd(initializer, caller);
        }
    }

protected:
    typedef void (*kernel)(const void*, const void*);

    inline void init_call_args(jit_snippets_call_args& call_args, const std::vector<MemoryPtr>& srcMemPtrs,
                               const std::vector<MemoryPtr>& dstMemPtrs, size_t ithr) {
        for (size_t i = 0; i < srcMemPtrs.size(); i++)
            call_args.src_ptrs[i] = srcMemPtrs[i]->getDataAs<const uint8_t>() + m_start_offset_in[i];

        for (size_t i = 0; i < dstMemPtrs.size(); i++)
            call_args.dst_ptrs[i] = dstMemPtrs[i]->getDataAs<uint8_t>() + m_start_offset_out[i];

        update_scratchpad_ptr(call_args.buffer_scratchpad_ptr, ithr);
    }
};

// Specialized dynamic executor based on shape agnostic kernel for the specific input shapes
class SubgraphDynamicSpecializedExecutor : public Subgraph::SubgraphExecutor {
public:
    SubgraphDynamicSpecializedExecutor(const std::shared_ptr<Subgraph::SubgraphAttrs>& snippet_attrs,
                                       const std::shared_ptr<Subgraph::SubgraphCodeGenerator>& snippet,
                                       const std::vector<ptrdiff_t>& start_offset_in,
                                       const std::vector<ptrdiff_t>& start_offset_out,
                                       const std::shared_ptr<CPURuntimeConfig>& snippet_config,
                                       const BufferScratchpadAllocator& allocator)
    : SubgraphExecutor(snippet_attrs, snippet, start_offset_in, start_offset_out, snippet_config, allocator) {
        buffer_offsets = snippet_config->buffer_cluster_offsets;
        data_offsets = snippet_config->io_data_offsets;
        loop_args = snippet_config->loop_args;
        reset_exec_table_state = snippet_config->kernel_executor_table->get_state_reset();
    }

    void exec(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) override {
        const auto& callable = m_schedule->get_callable<dynamic_kernel>();

        OPENVINO_ASSERT(data_offsets.size() == inMemPtrs.size() + outMemPtrs.size(), "Incorrect data offset count!");
        OPENVINO_ASSERT(data_offsets.front().size() == m_parallel_exec_domain.size(), "Data offsets with invalid ranks detected");

        // Note: we need to reset KernelExecutorTable to the state that was recorded in the SubgraphDynamicSpecializedExecutor
        // constructor because the table might've been used for other shapes
        reset_exec_table_state();

        std::vector<const uint8_t*> src_ptrs;
        std::vector<uint8_t*> dst_ptrs;
        init_original_ptrs(inMemPtrs, outMemPtrs, src_ptrs, dst_ptrs);

        auto initializer = [&](jit_snippets_call_args& call_args, size_t ithr) {
            init_call_args(call_args, ithr);
        };
        auto caller = [&](jit_snippets_call_args& call_args, const size_t* indexes) {
            update_ptrs(call_args, src_ptrs, dst_ptrs, indexes);
            callable(&call_args);
        };

        if (m_parallel_exec_domain.size() == rank6D) {
            parallel_for6d(initializer, caller);
        } else {
            parallel_forNd(initializer, caller);
        }
    }

protected:
    typedef void (*dynamic_kernel)(const void *);

    inline void init_call_args(jit_snippets_call_args& call_args, size_t ithr) {
        call_args.register_loops(loop_args);
        std::copy(buffer_offsets.cbegin(), buffer_offsets.cend(), call_args.buffer_offsets);

        update_scratchpad_ptr(call_args.buffer_scratchpad_ptr, ithr);
    }

    inline void init_original_ptrs(const std::vector<MemoryPtr>& srcMemPtrs, const std::vector<MemoryPtr>& dstMemPtrs,
                                   std::vector<const uint8_t*>& src_ptrs, std::vector<uint8_t*>& dst_ptrs) {
        const auto in_num = srcMemPtrs.size();
        const auto out_num = dstMemPtrs.size();

        src_ptrs.resize(in_num, nullptr);
        dst_ptrs.resize(out_num, nullptr);

        for (size_t i = 0; i < in_num; i++)
            src_ptrs[i] = srcMemPtrs[i]->getDataAs<const uint8_t>() + m_start_offset_in[i];
        for (size_t i = 0; i < out_num; i++)
            dst_ptrs[i] = dstMemPtrs[i]->getDataAs<uint8_t>() + m_start_offset_out[i];
    }

    inline void update_ptrs(jit_snippets_call_args& call_args, const std::vector<const uint8_t*>& src_ptrs,
                            const std::vector<uint8_t*>& dst_ptrs, const size_t* indexes) const {
        for (size_t i = 0; i < src_ptrs.size(); i++) {
            auto i_ptr = src_ptrs[i];
            for (size_t j = 0; j < data_offsets[i].size() - 1; j++) {
                i_ptr += data_offsets[i][j] * indexes[j];
            }
            call_args.src_ptrs[i] = i_ptr;
        }
        for (size_t i = 0; i < dst_ptrs.size(); i++) {
            auto i_ptr = dst_ptrs[i];
            for (size_t j = 0; j < data_offsets[i + src_ptrs.size()].size() - 1; j++) {
                i_ptr += data_offsets[i + src_ptrs.size()][j] * indexes[j];
            }
            call_args.dst_ptrs[i] = i_ptr;
        }
    }

    std::vector<size_t> buffer_offsets = {};
    std::vector<std::vector<size_t>> data_offsets = {};
    std::vector<jit_snippets_call_args::loop_args_t> loop_args = {};
    std::function<void()> reset_exec_table_state;
};

struct SubgraphKey {
    SubgraphKey() = default;
    SubgraphKey(const std::shared_ptr<Subgraph::SubgraphAttrs>& attrs_, const std::vector<VectorDims>& in_shapes_)
        : attrs(attrs_), in_shapes(in_shapes_) {}
    virtual ~SubgraphKey() = default;

    size_t hash() const;
    bool operator==(const SubgraphKey& rhs) const;

    std::shared_ptr<Subgraph::SubgraphAttrs> attrs = nullptr;
    std::vector<VectorDims> in_shapes = {};
};

struct SubgraphCodeGeneratorKey {
    SubgraphCodeGeneratorKey(const std::shared_ptr<Subgraph::SubgraphAttrs>& attrs_, uint8_t mask_)
        : attrs(attrs_), broadcasting_mask(mask_) {}

    size_t hash() const;
    bool operator==(const SubgraphCodeGeneratorKey& rhs) const;

    std::shared_ptr<Subgraph::SubgraphAttrs> attrs = nullptr;
    uint8_t broadcasting_mask = 0;
};

struct SubgraphShapeInferResultKey {
    SubgraphShapeInferResultKey(std::vector<VectorDims> in_shapes_, uint64_t body_hash_)
        : in_shapes(std::move(in_shapes_)), body_hash(body_hash_) {}

    size_t hash() const;
    bool operator==(const SubgraphShapeInferResultKey& rhs) const;

    std::vector<VectorDims> in_shapes = {};
    uint64_t body_hash = 0;
};

size_t get_attr_hash(size_t seed, const std::shared_ptr<Subgraph::SubgraphAttrs>& attrs) {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    for (const auto& order : attrs->inMemOrders)
        seed = get_vector_hash(seed, order);
    for (const auto& prec : attrs->inMemPrecs)
        seed = hash_combine(seed, prec.hash());

    for (const auto& order : attrs->outMemOrders)
        seed = get_vector_hash(seed, order);
    for (const auto& prec : attrs->outMemPrecs)
        seed = hash_combine(seed, prec.hash());

    seed = hash_combine(seed, attrs->bodyHash);
    return seed;
}


size_t SubgraphKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = get_attr_hash(0, attrs);
    for (const auto& shape : in_shapes)
        seed = get_vector_hash(seed, shape);

    return seed;
}

size_t SubgraphCodeGeneratorKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = get_attr_hash(0, attrs);
    seed = hash_combine(seed, broadcasting_mask);

    return seed;
}

size_t SubgraphShapeInferResultKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = hash_combine(0, body_hash);
    for (const auto& shape : in_shapes)
        seed = get_vector_hash(seed, shape);

    return seed;
}

bool operator==(const Subgraph::SubgraphAttrs& lhs, const Subgraph::SubgraphAttrs& rhs) {
    if (&lhs == &rhs)
        return true;
    if (lhs.bodyHash != rhs.bodyHash)
        return false;
    if (lhs.inMemOrders.size() != rhs.inMemOrders.size() ||
        lhs.inMemPrecs.size() != rhs.inMemPrecs.size())
        return false;
    if (lhs.outMemOrders.size() != rhs.outMemOrders.size() ||
        lhs.outMemPrecs.size() != rhs.outMemPrecs.size())
        return false;
    if (lhs.inMemOrders != rhs.inMemOrders ||
        lhs.inMemPrecs != rhs.inMemPrecs)
        return false;
    if (lhs.outMemOrders != rhs.outMemOrders ||
        lhs.outMemPrecs != rhs.outMemPrecs)
        return false;
    return true;
}

bool SubgraphKey::operator==(const SubgraphKey& rhs) const {
    return *attrs == *rhs.attrs && in_shapes == rhs.in_shapes;
}

bool SubgraphCodeGeneratorKey::operator==(const SubgraphCodeGeneratorKey& rhs) const {
    return *attrs == *rhs.attrs && broadcasting_mask == rhs.broadcasting_mask;
}

bool SubgraphShapeInferResultKey::operator==(const SubgraphShapeInferResultKey& rhs) const {
    return body_hash == rhs.body_hash && in_shapes == rhs.in_shapes;
}

struct SubgraphShapeInferResult {
    SubgraphShapeInferResult(IShapeInfer::Result res) : result(std::move(res)) {}

    IShapeInfer::Result result;
};

} // namespace

Subgraph::Subgraph(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
        : Node(op, context, SnippetShapeInferFactory(op)), subgraph_attrs(std::make_shared<SubgraphAttrs>()) {
#if defined(OPENVINO_ARCH_ARM64)
    host_isa = dnnl::impl::cpu::aarch64::asimd;
#else
    host_isa = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ? dnnl::impl::cpu::x64::avx512_core
                                                                                : dnnl::impl::cpu::x64::avx2;
#endif
    const auto& tmp_snippet = ov::as_type_ptr<snippets::op::Subgraph>(op);
    OPENVINO_ASSERT(tmp_snippet, "Attempt to create Subgraph node from an invalid op type");
    subgraph_attrs->snippet = tmp_snippet->clone();
    subgraph_attrs->bodyHash = getBodyHash(tmp_snippet);

#if defined(OPENVINO_ARCH_ARM64)
    subgraph_attrs->snippet->set_generator(std::make_shared<aarch64::CPUGenerator>(host_isa));
#elif defined(OPENVINO_ARCH_X86_64)
    subgraph_attrs->snippet->set_generator(std::make_shared<CPUGenerator>(host_isa, context->getParamsCache()));
#else
    OPENVINO_THROW("CPU plugin: Subgraphs code-generator is not supported on non-x64 platforms");
#endif

    // Note: we have to update shapeInfer, so it uses the per-thread op::Subgraph copy
    shapeInference = SnippetShapeInferFactory(subgraph_attrs->snippet).makeShapeInfer();
    is_dynamic = isDynamicNgraphNode(op);
}

uint64_t Subgraph::getBodyHash(const std::shared_ptr<snippets::op::Subgraph>& snippet) {
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
    // Domain sensitive operations and dynamic Subgraphs support only Planar layout
    const bool isOnlyPlanarApplicable = subgraph_attrs->snippet->has_domain_sensitive_ops();
    const bool isChannelsFirstApplicable = dnnl::impl::utils::one_of(ndims, 1u, 2u, 3u, 4u, 5u) && dimRanksAreEqual && !isOnlyPlanarApplicable && !isDynamic;
    // Todo: Subgraphs currently don't support per-channel broadcasting of Blocked descriptors because
    //  canonicalization can't distinguish between <N, C, H, W, c> and <N, C, D, H, W> cases.
    //  See snippets::op::Subgraph::canonicalize for details.
#if defined(OPENVINO_ARCH_ARM64)
    bool isBlockedApplicable = false;
#else
    bool isBlockedApplicable = dnnl::impl::utils::one_of(ndims,  3u, 4u, 5u) && dimRanksAreEqual && !isOnlyPlanarApplicable && !isDynamic;

    for (const auto& inShape : inputShapes) {
        if (isDynamic && inShape.getRank() != 1)
            isBlockedApplicable = isBlockedApplicable && inShape.getMinDims()[1] != Shape::UNDEFINED_DIM && inShape.getMinDims()[1] > 1;
    }
#endif

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
#if defined(OPENVINO_ARCH_ARM64)
                size_t blockSize = 16;
#else
                size_t blockSize = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ? 16 : 8;
#endif

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
                                     subgraph_attrs->snippet->has_domain_sensitive_ops()) ?
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
#if defined(OPENVINO_ARCH_ARM64)
        if (dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::asimd)) {
            impl_type = impl_desc_type::jit_asimd;
        }
#else
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
            impl_type = impl_desc_type::jit_avx512;
        } else if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
            impl_type = impl_desc_type::jit_avx2;
        }
#endif
        return {config, impl_type};
    };

    if (isChannelsFirstApplicable)
        supportedPrimitiveDescriptors.emplace_back(initDesc(ChannelsFirst));
    if (isBlockedApplicable)
        supportedPrimitiveDescriptors.emplace_back(initDesc(Blocked));
    supportedPrimitiveDescriptors.emplace_back(initDesc(Planar));
}

void Subgraph::selectOptimalPrimitiveDescriptor() {
    selectPreferPrimitiveDescriptorWithShape(getImplPriority(), true);
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

void Subgraph::createPrimitive() {
    if (!hasEmptyInputTensors()) {
        const auto config = getSelectedPrimitiveDescriptor()->getConfig();
        input_num = config.inConfs.size();
        output_num = config.outConfs.size();

        initMemoryPtrs();
        initPluginBlockedShapes();
        initAttributes();
        initStartOffsets();
        optimizeIR();
    }

    Node::createPrimitive();
}

void Subgraph::initMemoryPtrs() {
    srcMemPtrs.resize(input_num);
    dstMemPtrs.resize(output_num);
    for (size_t i = 0; i < input_num; i++)
        srcMemPtrs[i] = getSrcMemoryAtPort(i);
    for (size_t i = 0; i < output_num; i++)
        dstMemPtrs[i] = getDstMemoryAtPort(i);
}

void Subgraph::initAttributes() {
    const auto config = getSelectedPrimitiveDescriptor()->getConfig();

    subgraph_attrs->inMemPrecs.resize(input_num);
    subgraph_attrs->outMemPrecs.resize(output_num);

    subgraph_attrs->inMemOrders.resize(input_num);
    subgraph_attrs->outMemOrders.resize(output_num);

    for (size_t i = 0; i < input_num; i++) {
        const auto& memDesc = srcMemPtrs[i]->getDescWithType<BlockedMemoryDesc>();
        subgraph_attrs->inMemPrecs[i] = memDesc->getPrecision();
        subgraph_attrs->inMemOrders[i] = memDesc->getOrder();
    }
    for (size_t i = 0; i < output_num; i++) {
        const auto& memDesc = dstMemPtrs[i]->getDescWithType<BlockedMemoryDesc>();
        subgraph_attrs->outMemPrecs[i] = memDesc->getPrecision();
        subgraph_attrs->outMemOrders[i] = memDesc->getOrder();
    }
}

void Subgraph::initStartOffsets() {
    auto get_offset = [](const BlockedMemoryDescPtr& desc) {
        return static_cast<ptrdiff_t>(desc->getOffsetPadding() * desc->getPrecision().size());
    };
    start_offset_in.resize(input_num);
    start_offset_out.resize(output_num);
    for (size_t i = 0; i < input_num; i++)
        start_offset_in[i] = get_offset(srcMemPtrs[i]->getDescWithType<BlockedMemoryDesc>());
    for (size_t i = 0; i < output_num; i++)
        start_offset_out[i] = get_offset(dstMemPtrs[i]->getDescWithType<BlockedMemoryDesc>());
}

snippets::op::Subgraph::BlockedShapeVector Subgraph::getSnippetsBlockedShapes() const {
    const auto& config = getSelectedPrimitiveDescriptor()->getConfig();

    snippets::op::Subgraph::BlockedShapeVector in_blocked_shapes(input_num);
    for (size_t i = 0; i < input_num; i++) {
        const auto& memDesc = config.inConfs[i].getMemDesc();
        const auto& blockedDesc = memDesc->as<BlockedMemoryDesc>();
        const auto& order = blockedDesc->getOrder();

        in_blocked_shapes[i] = {blockedDesc->getBlockDims(), order};
    }
    return in_blocked_shapes;
}

std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>> Subgraph::getIOPrecisions() const {
    std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>> precisions;
    precisions.first.reserve(input_num);
    precisions.second.reserve(output_num);
    for (const auto& p : subgraph_attrs->inMemPrecs)
        precisions.first.push_back(p);
    for (const auto& p : subgraph_attrs->outMemPrecs)
        precisions.second.push_back(p);
    return precisions;
}

void Subgraph::initPluginBlockedShapes() const {
    in_shapes.resize(input_num);
    for (size_t i = 0; i < srcMemPtrs.size(); i++)
        in_shapes[i] = srcMemPtrs[i]->getDescWithType<BlockedMemoryDesc>()->getBlockDims();
}

Subgraph::DataFlowPasses Subgraph::getDataFlowPasses() {
    DataFlowPasses backend_passes;

    using PassPosition = ov::snippets::pass::PassPosition;
    using Place = PassPosition::Place;

#   define SNIPPETS_REGISTER_PASS_ABSOLUTE_COMMON(PASS_PLACE, PASS, ...) \
            backend_passes.emplace_back(PassPosition(PASS_PLACE), std::make_shared<PASS>(__VA_ARGS__))
#   define SNIPPETS_REGISTER_PASS_RELATIVE_COMMON(PASS_PLACE, TARGET_PASS, PASS, ...) \
            backend_passes.emplace_back(PassPosition(PASS_PLACE, TARGET_PASS::get_type_info_static()), std::make_shared<PASS>(__VA_ARGS__))

#if defined(OPENVINO_ARCH_X86_64)
#   define SNIPPETS_REGISTER_PASS_ABSOLUTE_X86_64(PASS_PLACE, PASS, ...) \
            backend_passes.emplace_back(PassPosition(PASS_PLACE), std::make_shared<PASS>(__VA_ARGS__))
#   define SNIPPETS_REGISTER_PASS_RELATIVE_X86_64(PASS_PLACE, TARGET_PASS, PASS, ...) \
            backend_passes.emplace_back(PassPosition(PASS_PLACE, TARGET_PASS::get_type_info_static()), std::make_shared<PASS>(__VA_ARGS__))
#else
#    define SNIPPETS_REGISTER_PASS_ABSOLUTE_X86_64(PASS_PLACE, PASS, ...)
#    define SNIPPETS_REGISTER_PASS_RELATIVE_X86_64(PASS_PLACE, TARGET_PASS, PASS, ...)
#endif  // OPENVINO_ARCH_X86_64

    SNIPPETS_REGISTER_PASS_ABSOLUTE_COMMON(Place::PipelineStart, ConvertToSwishCPU);
    SNIPPETS_REGISTER_PASS_RELATIVE_COMMON(Place::After, ov::snippets::pass::Canonicalization,
                                           ov::snippets::pass::AnalyzeBroadcastableInputs, broadcastable_inputs);
    if (context->getConfig().inferencePrecision == ov::element::bf16 && subgraph_attrs->snippet->has_domain_sensitive_ops()) {
        // enforce BF16 precisions to supported operations
        // MatMul has to be decomposed to Brgemm operations before enforcement
        // Note, MatMul decomposition will be run later again for case if BF16 enforcement is not happened
        SNIPPETS_REGISTER_PASS_ABSOLUTE_X86_64(Place::PipelineStart, ov::snippets::pass::MatMulToBrgemm);
        SNIPPETS_REGISTER_PASS_RELATIVE_X86_64(Place::After, ov::snippets::pass::MatMulToBrgemm,
                                               pass::EnforcePrecision, element::f32, element::bf16);
    }
    SNIPPETS_REGISTER_PASS_RELATIVE_X86_64(Place::Before, ov::snippets::pass::PropagatePrecision,
                                           ov::intel_cpu::pass::BrgemmToBrgemmCPU);
    SNIPPETS_REGISTER_PASS_ABSOLUTE_X86_64(Place::PipelineEnd, ov::intel_cpu::pass::RemoveConverts);
    SNIPPETS_REGISTER_PASS_ABSOLUTE_COMMON(Place::PipelineEnd, ov::intel_cpu::pass::MulAddToFMA);

#ifdef SNIPPETS_LIBXSMM_TPP
    SNIPPETS_REGISTER_PASS_RELATIVE_X86_64(Place::Before, ov::intel_cpu::pass::BrgemmToBrgemmCPU,
                                           ov::intel_cpu::tpp::pass::BrgemmToBrgemmTPP);
    // Note: There could be several ConvertConstantsToScalars instances in the pipeline
    SNIPPETS_REGISTER_PASS_ABSOLUTE_X86_64(Place::PipelineEnd, ov::intel_cpu::tpp::pass::ScalarToScalarTPP);
    SNIPPETS_REGISTER_PASS_RELATIVE_X86_64(Place::After, ov::intel_cpu::tpp::pass::BrgemmToBrgemmTPP,
                                           ov::intel_cpu::tpp::pass::EltwiseToEltwiseTPP);
    SNIPPETS_REGISTER_PASS_RELATIVE_X86_64(Place::After, ov::intel_cpu::tpp::pass::EltwiseToEltwiseTPP,
                                           ov::intel_cpu::tpp::pass::FuseTPPToEquations);
#endif

#undef SNIPPETS_REGISTER_PASS_ABSOLUTE_COMMON
#undef SNIPPETS_REGISTER_PASS_RELATIVE_COMMON
#undef SNIPPETS_REGISTER_PASS_ABSOLUTE_X86_64
#undef SNIPPETS_REGISTER_PASS_RELATIVE_X86_64

    return backend_passes;
}

Subgraph::ControlFlowPasses Subgraph::getControlFlowPasses() const {
    ControlFlowPasses backend_passes;

#if defined(OPENVINO_ARCH_X86_64)
    using PassPosition = ov::snippets::pass::PassPosition;
    using Place = PassPosition::Place;
#   define SNIPPETS_REGISTER_PASS_RELATIVE(PASS_PLACE, TARGET_PASS, PASS, ...) \
            backend_passes.emplace_back(PassPosition(PASS_PLACE, TARGET_PASS::get_type_info_static()), std::make_shared<PASS>(__VA_ARGS__))
#else
#    define SNIPPETS_REGISTER_PASS_RELATIVE(PASS_PLACE, TARGET_PASS, PASS, ...)
#endif  // OPENVINO_ARCH_X86_64

    SNIPPETS_REGISTER_PASS_RELATIVE(Place::After, ov::snippets::lowered::pass::MarkLoops,
                                    ov::intel_cpu::pass::BrgemmCPUBlocking);
    SNIPPETS_REGISTER_PASS_RELATIVE(Place::After, ov::snippets::lowered::pass::InsertLoops,
                                    ov::intel_cpu::pass::FuseLoadStoreConvert);
    SNIPPETS_REGISTER_PASS_RELATIVE(Place::Before, ov::snippets::lowered::pass::InsertBuffers,
                                    ov::intel_cpu::pass::InsertBrgemmCopyBBuffers);

#ifdef SNIPPETS_LIBXSMM_TPP
    SNIPPETS_REGISTER_PASS_RELATIVE(Place::Before, ov::intel_cpu::pass::BrgemmCPUBlocking,
                                    ov::intel_cpu::tpp::pass::BrgemmTPPBlocking);
    SNIPPETS_REGISTER_PASS_RELATIVE(Place::After, ov::intel_cpu::pass::FuseLoadStoreConvert,
                                    ov::intel_cpu::tpp::pass::SetTPPLeadingDim);
#endif

#undef SNIPPETS_REGISTER_PASS_RELATIVE
    return backend_passes;
}

uint8_t Subgraph::getBroadcastingMask(const std::vector<VectorDims>& input_shapes) {
    uint8_t mask = 0;
    for (const auto& broadcastable_input : broadcastable_inputs) {
        const auto& shape = input_shapes[broadcastable_input.first];
        mask = mask << 1;
        if (*(shape.rbegin() + broadcastable_input.second) == 1)
            mask = mask | 1;
    }
    return mask;
}

void Subgraph::optimizeIR() {
    const auto& subgraph = subgraph_attrs->snippet;

    const auto in_blocked_shapes = getSnippetsBlockedShapes();
    const auto precisions = getIOPrecisions();
    subgraph->data_flow_transformations(in_blocked_shapes, precisions.first, precisions.second, getDataFlowPasses());

    // DataFlow transformations includes AnalyzeBroadcastableInputs pass:
    // we should verify that the received map is aligned with our blocked input shapes
    OPENVINO_ASSERT((broadcastable_inputs.size() < in_shapes.size()) ||
                    (!broadcastable_inputs.empty() && broadcastable_inputs.rbegin()->first < in_shapes.size()),
                    "Incorrect indexes of broadcastable inputs of Subgraph");
    for (const auto broadcastable_input : broadcastable_inputs) {
        OPENVINO_ASSERT(broadcastable_input.second < in_shapes[broadcastable_input.first].size(),
                        "Incorrect processing dimension index of broadcastable index");
    }

    // TODO: Snippets don't support backend-provided blocking, so we need to reshape body
    //       using blocked shapes first. This can be removed after [121670]
    std::vector<snippets::VectorDimsRef> in_shapes;
    for (const auto& s : in_blocked_shapes)
        in_shapes.emplace_back(s.first);
    subgraph->shape_infer(in_shapes);

    const auto control_flow_config = std::make_shared<ov::snippets::lowered::pass::PassConfig>();
    const auto control_flow_passes = getControlFlowPasses();

#ifdef SNIPPETS_LIBXSMM_TPP
    // Note: temporary disabled. Re-enable after ticket 132833 is resolved
    control_flow_config->disable<ov::snippets::lowered::pass::OptimizeDomain>();

    subgraph->set_tile_rank(std::min(2ul, subgraph->infer_master_shape().size()));
#endif

    // Note: minimal JIT work amount is a predefined value that describes the number of kernel iterations (work amount)
    // needed to cover kernel call overhead. It is used for balancing between parallel and JIT work amounts in domain optimization.
    subgraph->control_flow_transformations(static_cast<size_t>(parallel_get_max_threads()), 256,
                                           std::make_shared<snippets::CPUShapeInferSnippetsFactory>(),
                                           control_flow_config, control_flow_passes);
}

void Subgraph::prepareParams() {
    const auto& cache = context->getParamsCache();

    auto builder = [this, &cache](const SubgraphKey& key) -> std::shared_ptr<SubgraphExecutor> {
        const auto& snippet = subgraph_attrs->snippet;

        SubgraphExecutor::BufferScratchpadAllocator allocator = [this](size_t size) {
            return getScratchPadMem(std::make_shared<CpuBlockedMemoryDesc>(ov::element::u8, intel_cpu::Shape{size}));
        };

        if (is_dynamic) {
            // Dynamic case:
            // 1. Generate JIT code if needed
            // 2. Update runtime config with dynamic values
            //    If JIT code has been taken from cache, need to set cached kernel executor table for the configuration
            // 3. Create SubgraphDynamicSpecializedExecutor
            const auto code_gen_result = cache->getOrCreate(SubgraphCodeGeneratorKey(subgraph_attrs, getBroadcastingMask(in_shapes)),
                                                            [](const SubgraphCodeGeneratorKey& key) -> std::shared_ptr<SubgraphCodeGenerator> {
                                                                return std::make_shared<SubgraphCodeGenerator>(key.attrs, std::make_shared<CPURuntimeConfig>());
                                                            });
            const auto& code_gen = code_gen_result.first;
            // [148644] : Update Kernel table from SubgraphCodeGenerator when JIT code was already generated with specific Kernel table
            if (code_gen_result.second == CacheEntryBase::LookUpStatus::Hit) {
                snippet->get_runtime_configurator()->set_kernel_executor_table(code_gen->get()->lowering_result.kernel_executor_table);
            }
            const auto& snippet_config = ov::as_type_ptr<CPURuntimeConfig>(snippet->update_runtime_config());
            return std::make_shared<SubgraphDynamicSpecializedExecutor>(key.attrs, code_gen, start_offset_in, start_offset_out, snippet_config, allocator);
        } else {
            // Static case:
            // 1. Update runtime config to get static scheduling data (io data offsets, parallel domain) which will be compiled in JIT code
            // 2. Generate JIT code with this static data if needed
            // 3. Create SubgraphStaticExecutor
            const auto& snippet_config = ov::as_type_ptr<CPURuntimeConfig>(snippet->update_runtime_config());
            const auto code_gen_result = cache->getOrCreate(SubgraphCodeGeneratorKey(subgraph_attrs, getBroadcastingMask(in_shapes)),
                                                            [&snippet_config](const SubgraphCodeGeneratorKey& key) -> std::shared_ptr<SubgraphCodeGenerator> {
                                                                return std::make_shared<SubgraphCodeGenerator>(key.attrs, snippet_config);
                                                            });
            return std::make_shared<SubgraphStaticExecutor>(key.attrs, code_gen_result.first, start_offset_in, start_offset_out, snippet_config, allocator);
        }
    };

    const auto result = cache->getOrCreate(SubgraphKey(subgraph_attrs, in_shapes), builder);
    execPtr = result.first;
    OPENVINO_ASSERT(execPtr != nullptr, "Executor is not created for node ", getName(), ".");
}

IShapeInfer::Result Subgraph::shapeInfer() const {
    for (size_t i = 0; i < srcMemPtrs.size(); i++)
        in_shapes[i] = srcMemPtrs[i]->getDescWithType<BlockedMemoryDesc>()->getBlockDims();

    auto builder = [this](const SubgraphShapeInferResultKey& key) -> std::shared_ptr<SubgraphShapeInferResult> {
        return std::make_shared<SubgraphShapeInferResult>(Node::shapeInfer());
    };

    const auto cache = context->getParamsCache();
    const auto result = cache->getOrCreate(SubgraphShapeInferResultKey(in_shapes, subgraph_attrs->bodyHash), builder);
    return result.first->result;
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
    OPENVINO_ASSERT(execPtr, "Can't execute Subgraph node. Primitive didn't created");
    execPtr->exec(srcMemPtrs, dstMemPtrs);
}

void Subgraph::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

namespace {
inline void init_parallel_domain(const std::shared_ptr<CPURuntimeConfig>& snippet_config, std::vector<size_t>& domain) {
    const auto& master_shape = snippet_config->master_shape;
    const auto& tensor_rank = snippet_config->tensor_rank;
    const auto& tile_rank = snippet_config->tile_rank;
    domain.resize(tensor_rank, 1);

    std::fill(domain.begin(), domain.end(), 1);
    std::copy(master_shape.cbegin(), master_shape.cbegin() + (master_shape.size() - tile_rank),
              domain.begin() + (tensor_rank - master_shape.size()));
}
}  // namespace

Subgraph::SubgraphCodeGenerator::SubgraphCodeGenerator(const std::shared_ptr<Subgraph::SubgraphAttrs>& snippet_attrs,
                                                       const std::shared_ptr<CPURuntimeConfig>& config) {
    OPENVINO_ASSERT(snippet_attrs, "Subgraph attributes are empty!");
    OPENVINO_ASSERT(config, "Runtime Config is empty!");

    jit_snippets_compile_args jcp;
    jcp.data_offsets = config->io_data_offsets;
    init_parallel_domain(config, jcp.exec_domain);
    schedule = std::make_shared<ov::snippets::Schedule>(snippet_attrs->snippet->generate(reinterpret_cast<const void*>(&jcp)));
}

Subgraph::SubgraphExecutor::SubgraphExecutor(const std::shared_ptr<Subgraph::SubgraphAttrs>& snippet_attrs,
                                             const std::shared_ptr<SubgraphCodeGenerator>& snippet,
                                             const std::vector<ptrdiff_t>& start_offset_in,
                                             const std::vector<ptrdiff_t>& start_offset_out,
                                             const std::shared_ptr<CPURuntimeConfig>& snippet_config,
                                             const BufferScratchpadAllocator& allocator)
    : m_schedule(snippet->get()), m_start_offset_in(start_offset_in), m_start_offset_out(start_offset_out) {
    OPENVINO_ASSERT(m_schedule, "Schedule is empty!");
    OPENVINO_ASSERT(snippet_config, "Runtime Config is empty!");
    init_parallel_domain(snippet_config, m_parallel_exec_domain);

    m_harness_work_amount = std::accumulate(m_parallel_exec_domain.cbegin(), m_parallel_exec_domain.cend(), size_t(1), std::multiplies<size_t>());
    m_nthreads = std::min(parallel_get_max_threads(), static_cast<int>(m_harness_work_amount));

    m_buffer_scratchpad_size = snippet_config->buffer_scratchpad_size;
    OPENVINO_ASSERT(!ov::snippets::utils::is_dynamic_value(m_buffer_scratchpad_size), "Undefined buffer scratchpad size!");
    m_buffer_scratchpad = allocator(static_cast<size_t>(m_nthreads) * m_buffer_scratchpad_size);

#if defined(__linux__) && defined(OPENVINO_ARCH_X86_64) && defined(SNIPPETS_DEBUG_CAPS)
    const auto target = std::dynamic_pointer_cast<const CPUTargetMachine>(snippet_attrs->snippet->get_generator()->get_target_machine());
    enabled_segfault_detector = target && target->debug_config.enable_segfault_detector;
#endif
}

#if defined(__linux__) && defined(OPENVINO_ARCH_X86_64) && defined(SNIPPETS_DEBUG_CAPS)
void Subgraph::SubgraphExecutor::segfault_detector() {
    if (enabled_segfault_detector) {
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

void Subgraph::SubgraphExecutor::parallel_for6d(const std::function<void(jit_snippets_call_args&, size_t)>& initializer,
                                                const std::function<void(jit_snippets_call_args&, const size_t*)>& caller) {
    const auto& dom = m_parallel_exec_domain;

#if defined(__linux__) && defined(OPENVINO_ARCH_X86_64) && defined(SNIPPETS_DEBUG_CAPS)
    segfault_detector();
#endif

    parallel_nt_static(m_nthreads, [&](const int ithr, const int nthr) {
        jit_snippets_call_args call_args;
        initializer(call_args, ithr);

        size_t start = 0, end = 0;
        splitter(m_harness_work_amount, nthr, ithr, start, end);

        size_t indexes[] = {0, 0, 0, 0, 0};
        parallel_it_init(start, indexes[0], dom[0], indexes[1], dom[1], indexes[2], dom[2], indexes[3], dom[3], indexes[4], dom[4]);
        for (size_t iwork = start; iwork < end; ++iwork) {
            caller(call_args, indexes);
            parallel_it_step(indexes[0], dom[0], indexes[1], dom[1], indexes[2], dom[2], indexes[3], dom[3], indexes[4], dom[4]);
        }
    });
}

void Subgraph::SubgraphExecutor::parallel_forNd(const std::function<void(jit_snippets_call_args&, size_t)>& initializer,
                                                const std::function<void(jit_snippets_call_args&, const size_t*)>& caller) {
    const auto& dom = m_parallel_exec_domain;

#if defined(__linux__) && defined(OPENVINO_ARCH_X86_64) && defined(SNIPPETS_DEBUG_CAPS)
    segfault_detector();
#endif

    parallel_nt_static(m_nthreads, [&](const int ithr, const int nthr) {
        jit_snippets_call_args call_args;
        initializer(call_args, ithr);

        size_t start = 0, end = 0;
        splitter(m_harness_work_amount, nthr, ithr, start, end);

        std::vector<size_t> indexes(dom.size() - 1, 0);
        for (size_t iwork = start; iwork < end; ++iwork) {
            size_t tmp = iwork;
            for (ptrdiff_t j = static_cast<ptrdiff_t>(dom.size()) - 2; j >= 0; j--) {
                indexes[j] = tmp % dom[j];
                tmp /= dom[j];
            }

            caller(call_args, indexes.data());
        }
    });
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
