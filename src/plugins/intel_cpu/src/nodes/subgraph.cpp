// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "subgraph.h"

#include "snippets/op/subgraph.hpp"
#include "snippets/pass/hash.hpp"
#include "snippets/pass/matmul_to_brgemm.hpp"
#include "snippets/pass/propagate_precision.hpp"
#include "snippets/pass/positioned_pass.hpp"
#include "snippets/lowered/pass/pass.hpp"
#include "snippets/lowered/pass/insert_broadcastmove.hpp"
#include "snippets/lowered/pass/load_movebroadcast_to_broadcastload.hpp"
#include "snippets/lowered/pass/insert_loops.hpp"
#include "snippets/lowered/pass/mark_loops.hpp"
#include "snippets/lowered/pass/set_load_store_scalar.hpp"

#include "emitters/snippets/x64/cpu_generator.hpp"

#include "transformations/defs.hpp"
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

#include "shape_inference/custom/subgraph.hpp"
#include "utils/cpu_utils.hpp"
#include "common/primitive_hashing_utils.hpp"

#include "openvino/core/parallel.hpp"


namespace ov {
namespace intel_cpu {
namespace node {

namespace {

struct SnippetKey {
    Subgraph::SnippetAttrs attrs;

    size_t hash() const;
    bool operator==(const SnippetKey& rhs) const;
};

size_t SnippetKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    for (const auto& order : attrs.inMemOrders)
        seed = get_vector_hash(seed, order);
    for (const auto& prec : attrs.inMemPrecs)
        seed = hash_combine(seed, prec.hash());

    for (const auto& order : attrs.outMemOrders)
        seed = get_vector_hash(seed, order);
    for (const auto& prec : attrs.outMemPrecs)
        seed = hash_combine(seed, prec.hash());

    seed = hash_combine(seed, attrs.bodyHash);
    seed = hash_combine(seed, attrs.broadcasting_mask);

    return seed;
}

bool SnippetKey::operator==(const SnippetKey& rhs) const {
    if (attrs.bodyHash != rhs.attrs.bodyHash ||
        attrs.broadcasting_mask != rhs.attrs.broadcasting_mask)
        return false;
    if (attrs.inMemOrders.size() != rhs.attrs.inMemOrders.size() ||
        attrs.inMemPrecs.size() != rhs.attrs.inMemPrecs.size())
        return false;
    if (attrs.outMemOrders.size() != rhs.attrs.outMemOrders.size() ||
        attrs.outMemPrecs.size() != rhs.attrs.outMemPrecs.size())
        return false;

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
                size_t blockSize = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ? 16 : 8;

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
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
            impl_type = impl_desc_type::jit_avx512;
        } else if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
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

void Subgraph::createPrimitive() {
    const auto config = getSelectedPrimitiveDescriptor()->getConfig();
    input_num = config.inConfs.size();
    output_num = config.outConfs.size();

    init_memory_ptrs();
    init_attrs();
    init_start_offsets();
    lower();

    Node::createPrimitive();
}

void Subgraph::init_memory_ptrs() {
    srcMemPtrs.resize(input_num);
    dstMemPtrs.resize(output_num);
    for (size_t i = 0; i < input_num; i++)
        srcMemPtrs[i] = getParentEdgeAt(i)->getMemoryPtr();
    for (size_t i = 0; i < output_num; i++)
        dstMemPtrs[i] = getChildEdgeAt(i)->getMemoryPtr();
}

void Subgraph::init_attrs() {
    const auto config = getSelectedPrimitiveDescriptor()->getConfig();

    snippetAttrs.inMemPrecs.resize(input_num);
    snippetAttrs.outMemPrecs.resize(output_num);

    snippetAttrs.inMemOrders.resize(input_num);
    snippetAttrs.outMemOrders.resize(output_num);

    snippetAttrs.has_non_planar_inputs = false;

    for (size_t i = 0; i < input_num; i++) {
        const auto& memDesc = config.inConfs[i].getMemDesc();
        snippetAttrs.inMemPrecs[i] = memDesc->getPrecision();
        snippetAttrs.inMemOrders[i] = memDesc->as<BlockedMemoryDesc>()->getOrder();
        snippetAttrs.has_non_planar_inputs |= !memDesc->hasLayoutType(LayoutType::ncsp);
    }
    for (size_t i = 0; i < output_num; i++) {
        const auto& memDesc = config.outConfs[i].getMemDesc();
        snippetAttrs.outMemPrecs[i] = memDesc->getPrecision();
        snippetAttrs.outMemOrders[i] = memDesc->as<BlockedMemoryDesc>()->getOrder();
    }
}

void Subgraph::init_start_offsets() {
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

void Subgraph::init_snippets_blocked_shapes(snippets::op::Subgraph::BlockedShapeVector& in_blocked_shapes) {
    const auto config = getSelectedPrimitiveDescriptor()->getConfig();

    in_blocked_shapes.reserve(input_num);
    for (size_t i = 0; i < input_num; i++) {
        const auto& memDesc = config.inConfs[i].getMemDesc();
        const auto& blockedDesc = memDesc->as<BlockedMemoryDesc>();
        const auto& order = blockedDesc->getOrder();

        in_blocked_shapes.emplace_back(blockedDesc->getBlockDims(), order);
    }
}

void Subgraph::init_precisions(std::vector<ov::element::Type>& input_types, std::vector<ov::element::Type>& output_types) {
    input_types.reserve(input_num);
    output_types.reserve(output_num);
    for (const auto& p : snippetAttrs.inMemPrecs)
        input_types.push_back(p);
    for (const auto& p : snippetAttrs.outMemPrecs)
        output_types.push_back(p);
}

void Subgraph::lower() {
    snippets::op::Subgraph::BlockedShapeVector in_blocked_shapes;
    std::vector<ov::element::Type> input_precisions, output_precisions;
    init_snippets_blocked_shapes(in_blocked_shapes);
    init_precisions(input_precisions, output_precisions);

    const auto data_flow_backend_passes = get_data_flow_passes();
    const auto control_flow_backend_passes = get_control_flow_passes();

    snippetAttrs.snippet->data_flow_transformations(in_blocked_shapes, input_precisions, output_precisions, data_flow_backend_passes);
    snippetAttrs.snippet->convert_body_to_linear_ir(std::make_shared<snippets::CPUShapeInferSnippetsFactory>());

    // todo: snippets don't support backend-provided blocking, so we need to reshape body
    //  using blocked shapes first. This can be removed after [121670]
    // if (snippetAttrs.has_non_planar_inputs) {
    //     std::vector<snippets::VectorDimsRef> in_shapes;
    //     for (const auto& s : snippetAttrs.inMemBlockedDims)
    //         in_shapes.emplace_back(s);
    //     snippetAttrs.snippet->shape_infer(in_shapes);
    // }
    //const VectorDims& canonicalShape = {};

    // initialize by maximum output dimension. Dimensions of outputs should be broadcastable
    tensor_rank = rank6D; // std::max(static_cast<size_t>(rank6D), canonicalShape.size());
    snippetAttrs.snippet->set_tensor_rank(tensor_rank);
    snippetAttrs.snippet->set_min_parallel_work_amount(static_cast<size_t>(parallel_get_max_threads()));
    // Note: minimal JIT work amount is a predefined value that describes the number of kernel iterations (work amount)
    // needed to cover kernel call overhead. It is used for balancing between parallel and JIT work amounts in domain optimization.
    snippetAttrs.snippet->set_min_jit_work_amount(256);
    snippetAttrs.snippet->control_flow_transformations(control_flow_backend_passes, std::make_shared<ov::snippets::lowered::pass::PassConfig>());
}

std::vector<ov::snippets::pass::Manager::PositionedPassBase> Subgraph::get_data_flow_passes() const {
    std::vector<ov::snippets::pass::Manager::PositionedPassBase> backend_passes;
#if defined(OPENVINO_ARCH_X86_64)
    using PassPosition = ov::snippets::pass::PassPosition;
    using Place = PassPosition::Place;
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
        SNIPPETS_REGISTER_PASS(PassPosition(Place::After, ov::snippets::pass::MatMulToBrgemm::get_type_info_static()),
                               pass::EnforcePrecision, element::f32, element::bf16);
    }
    SNIPPETS_REGISTER_PASS(PassPosition(Place::Before, ov::snippets::pass::PropagatePrecision::get_type_info_static()),
                           ov::intel_cpu::pass::BrgemmToBrgemmCPU);
    SNIPPETS_REGISTER_PASS(PassPosition(Place::After, ov::intel_cpu::pass::BrgemmToBrgemmCPU::get_type_info_static()),
                           ov::intel_cpu::pass::SetBrgemmCPUBlockingParams);
    SNIPPETS_REGISTER_PASS(PassPosition(Place::PipelineEnd), ov::intel_cpu::pass::RemoveConverts);
    SNIPPETS_REGISTER_PASS(PassPosition(Place::PipelineEnd), ov::intel_cpu::pass::MulAddToFMA);

#undef SNIPPETS_REGISTER_PASS
    return backend_passes;
}

std::vector<ov::snippets::lowered::pass::PassPipeline::PositionedPassLowered> Subgraph::get_control_flow_passes() const {
    std::vector<ov::snippets::lowered::pass::PassPipeline::PositionedPassLowered> backend_passes;
#if defined(OPENVINO_ARCH_X86_64)
    using PassPosition = ov::snippets::pass::PassPosition;
    using Place = PassPosition::Place;
#   define SNIPPETS_REGISTER_PASS(PASS_POS, PASS, ...) \
        backend_passes.emplace_back(PASS_POS, std::make_shared<PASS>(__VA_ARGS__))
#else
#    define SNIPPETS_REGISTER_PASS(PASS_POS, PASS, ...)
#endif  // OPENVINO_ARCH_X86_64

    SNIPPETS_REGISTER_PASS(PassPosition(Place::After, ov::snippets::lowered::pass::MarkLoops::get_type_info_static()),
                           ov::intel_cpu::pass::BrgemmBlocking);
    SNIPPETS_REGISTER_PASS(PassPosition(Place::After, ov::snippets::lowered::pass::InsertLoops::get_type_info_static()),
                           ov::intel_cpu::pass::FuseLoadStoreConvert);
    SNIPPETS_REGISTER_PASS(PassPosition(Place::After, ov::intel_cpu::pass::FuseLoadStoreConvert::get_type_info_static()),
                           ov::intel_cpu::pass::SetBrgemmCopyBBuffersShape);

#undef SNIPPETS_REGISTER_PASS
    return backend_passes;
}

bool Subgraph::needPrepareParams() const {
    return inputShapesModified();
}

void Subgraph::prepareParams() {
    SnippetKey key = {snippetAttrs};
    key.attrs.broadcasting_mask = get_blocked_broadcasting_mask();

    auto builder = [this](const SnippetKey& key) -> std::shared_ptr<SnippetExecutor> {
        std::shared_ptr<SnippetExecutor> executor =
                std::make_shared<SnippetJitExecutor>(key.attrs, is_dynamic, tensor_rank, start_offset_in, start_offset_out);
        return executor;
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);
    execPtr = result.first;
    OPENVINO_ASSERT(execPtr != nullptr, "Executor is not created for node ", getName(), ".");
    execPtr->prepare();
}

uint8_t Subgraph::get_blocked_broadcasting_mask() {
    uint8_t mask = 0;
    // TODO: add check for non-eltwise inputs
    for (const auto& memptr : srcMemPtrs) {
        mask = mask << 1;
        if (memptr->getDescWithType<BlockedMemoryDesc>()->getBlockDims().back() == 1)
            mask = mask | 1;
    }
    return mask;
}

void Subgraph::execute(dnnl::stream strm) {
    OPENVINO_ASSERT(execPtr != nullptr, "Executor is not ready for node ", getName(), " for inference.");
    execPtr->exec(srcMemPtrs, dstMemPtrs);
}

void Subgraph::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
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

bool Subgraph::canBeInPlace() const {
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

bool Subgraph::created() const {
    return getType() == Type::Subgraph;
}

Subgraph::SnippetExecutor::SnippetExecutor(SnippetAttrs attrs, bool is_dynamic, size_t tensor_rank,
                                           const std::vector<ptrdiff_t>& start_offset_in, const std::vector<ptrdiff_t>& start_offset_out)
    : snippet_attrs(std::move(attrs)), is_dynamic(is_dynamic), tensor_rank(tensor_rank), start_offset_in(start_offset_in), start_offset_out(start_offset_out) {}

Subgraph::SnippetJitExecutor::SnippetJitExecutor(SnippetAttrs attrs, bool is_dynamic, size_t tensor_rank,
                                                 const std::vector<ptrdiff_t>& start_offset_in, const std::vector<ptrdiff_t>& start_offset_out)
    : SnippetExecutor(attrs, is_dynamic, tensor_rank, start_offset_in, start_offset_out) {
    ov::snippets::lowered::pass::PassPipeline custom_control_flow_pipeline;
    custom_control_flow_pipeline.register_pass<ov::snippets::lowered::pass::SetLoadStoreScalar>();
    custom_control_flow_pipeline.register_pass<ov::snippets::lowered::pass::InsertBroadcastMove>();
    custom_control_flow_pipeline.register_pass<ov::snippets::lowered::pass::LoadMoveBroadcastToBroadcastLoad>();

    jit_snippets_compile_args jcp;
    jcp.parallel_executor_ndims = tensor_rank;
    schedule = snippet_attrs.snippet->generate_from_linear_ir(custom_control_flow_pipeline, reinterpret_cast<const void*>(&jcp));
}

void Subgraph::SnippetJitExecutor::prepare() {
    buffer_scratchpad_size = schedule.lowering_result.buffer_scratchpad_size;
    buffer_scratchpad.resize(buffer_scratchpad_size * parallel_get_max_threads(), 0);
    parallel_exec_domain = snippet_attrs.snippet->get_parallel_exec_domain();
}

void Subgraph::SnippetJitExecutor::exec(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) {
    if (tensor_rank == rank6D) {
        schedule_6d(inMemPtrs, outMemPtrs);
    } else {
        schedule_nt(inMemPtrs, outMemPtrs);
    }
}

inline void Subgraph::SnippetJitExecutor::update_ptrs(jit_snippets_call_args& call_args,
                                                      const std::vector<MemoryPtr>& srcMemPtrs,
                                                      const std::vector<MemoryPtr>& dstMemPtrs) {
    for (size_t i = 0; i < srcMemPtrs.size(); i++)
        call_args.src_ptrs[i] = reinterpret_cast<const uint8_t*>(srcMemPtrs[i]->getData()) + start_offset_in[i];

    for (size_t i = 0; i < dstMemPtrs.size(); i++)
        call_args.dst_ptrs[i] = reinterpret_cast<uint8_t*>(dstMemPtrs[i]->getData()) + start_offset_out[i];

    if (buffer_scratchpad_size > 0) {
        call_args.buffer_scratchpad_ptr =
                reinterpret_cast<uint8_t*>(buffer_scratchpad.data()) + parallel_get_thread_num() * buffer_scratchpad_size;
    }
}

inline void Subgraph::SnippetJitExecutor::update_ptrs(jit_snippets_call_args& call_args,
                                                      const std::vector<MemoryPtr>& srcMemPtrs,
                                                      const std::vector<MemoryPtr>& dstMemPtrs,
                                                      const int64_t* indexes,
                                                      const std::vector<std::vector<int64_t>>& data_offsets) {
    OPENVINO_ASSERT(data_offsets.size() == srcMemPtrs.size() + dstMemPtrs.size(), "Incorrect data offset count!");
    OPENVINO_ASSERT(data_offsets.front().size() == tensor_rank, "Data offsets with invalid ranks detected");

    for (size_t i = 0; i < srcMemPtrs.size(); i++) {
        auto i_ptr = reinterpret_cast<uint8_t*>(srcMemPtrs[i]->getData()) + start_offset_in[i];
        for (size_t j = 0; j < tensor_rank - 1; j++) {
            i_ptr += (data_offsets[i][j] * indexes[j]);
        }
        call_args.src_ptrs[i] = i_ptr;
    }
    for (size_t i = 0; i < dstMemPtrs.size(); i++) {
        auto i_ptr = reinterpret_cast<uint8_t*>(dstMemPtrs[i]->getData()) + start_offset_out[i];
        for (size_t j = 0; j < tensor_rank - 1; j++) {
            i_ptr += (data_offsets[i + srcMemPtrs.size()][j] * indexes[j]);
        }
        call_args.dst_ptrs[i] = i_ptr;
    }
    if (buffer_scratchpad_size > 0) {
        call_args.buffer_scratchpad_ptr =
                reinterpret_cast<uint8_t*>(buffer_scratchpad.data()) + parallel_get_thread_num() * buffer_scratchpad_size;
    }
}

void Subgraph::SnippetJitExecutor::schedule_6d(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) {
    const auto& dom = parallel_exec_domain;
    OPENVINO_ASSERT(dom.size() == tensor_rank, "Incorrect parallel execution domain rank!");

    if (is_dynamic) {
        const auto& callable = schedule.get_callable<dynamic_kernel>();
        parallel_for5d(dom[0], dom[1], dom[2], dom[3], dom[4],
            [&](int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4) {
                int64_t indexes[] = {d0, d1, d2, d3, d4};
                // todo: jit_snippets_call_args are destructed at the end of this lambda.
                //  It means that rather expensive memory allocation-deallocation is performed inside this loop.
                //  A possible solution is to create thread-local jit_snippets_call_args that would be reused here.
                jit_snippets_call_args call_args;
                call_args.register_loops(loop_args);
                update_ptrs(call_args, inMemPtrs, outMemPtrs, indexes, data_offsets);
                callable(&call_args);
        });
    } else {
        const auto& callable = schedule.get_callable<kernel>();
        parallel_for5d(dom[0], dom[1], dom[2], dom[3], dom[4],
            [&](int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4) {
                int64_t indexes[] = {d0, d1, d2, d3, d4};
                jit_snippets_call_args call_args;
                update_ptrs(call_args, inMemPtrs, outMemPtrs);
                callable(indexes, &call_args);
        });
    }
}

void Subgraph::SnippetJitExecutor::schedule_nt(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) {
    const auto& work_size = parallel_exec_domain;
    OPENVINO_ASSERT(work_size.size() == tensor_rank, "Incorrect parallel execution domain rank!");

    if (is_dynamic) {
        parallel_nt(0, [&](const int ithr, const int nthr) {
            jit_snippets_call_args call_args;

            size_t start = 0, end = 0;
            splitter(harness_work_amount, nthr, ithr, start, end);

            std::vector<int64_t> indexes(work_size.size() - 1, 0);
            for (size_t iwork = start; iwork < end; ++iwork) {
                size_t tmp = iwork;
                for (ptrdiff_t j = static_cast<ptrdiff_t>(work_size.size()) - 2; j >= 0; j--) {
                    indexes[j] = static_cast<int64_t>(tmp % work_size[j]);
                    tmp /= work_size[j];
                }

                update_ptrs(call_args, inMemPtrs, outMemPtrs, indexes.data(), data_offsets);
                schedule.get_callable<kernel>()(indexes.data(), &call_args);
            }
        });

    } else {
        parallel_nt(0, [&](const int ithr, const int nthr) {
            jit_snippets_call_args call_args;
            update_ptrs(call_args, inMemPtrs, outMemPtrs);

            size_t start = 0, end = 0;
            splitter(harness_work_amount, nthr, ithr, start, end);

            std::vector<int64_t> indexes(work_size.size() - 1, 0);
            for (size_t iwork = start; iwork < end; ++iwork) {
                size_t tmp = iwork;
                for (ptrdiff_t j = static_cast<ptrdiff_t>(work_size.size()) - 2; j >= 0; j--) {
                    indexes[j] = static_cast<int64_t>(tmp % work_size[j]);
                    tmp /= work_size[j];
                }

                schedule.get_callable<kernel>()(indexes.data(), &call_args);
            }
        });
    }
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
