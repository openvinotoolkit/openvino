// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "subgraph.h"

#include "common/primitive_hashing_utils.hpp"
#include "dnnl_extension_utils.h"
#include "onednn/dnnl.h"
#include "openvino/core/parallel.hpp"
#include "shape_inference/custom/subgraph.hpp"
#include "snippets/lowered/pass/init_loops.hpp"
#include "snippets/lowered/pass/insert_buffers.hpp"
#include "snippets/lowered/pass/insert_loops.hpp"
#include "snippets/lowered/pass/insert_perf_count_verbose.hpp"
#include "snippets/lowered/pass/mark_loops.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/pass/analyze_broadcastable_inputs.hpp"
#include "snippets/pass/canonicalization.hpp"
#include "snippets/pass/hash.hpp"
#include "snippets/pass/matmul_to_brgemm.hpp"
#include "snippets/pass/positioned_pass.hpp"
#include "snippets/pass/propagate_precision.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/cpu_opset/common/pass/convert_to_swish_cpu.hpp"
#include "transformations/defs.hpp"
#include "transformations/snippets/common/pass/mul_add_to_fma.hpp"

#if defined(OPENVINO_ARCH_ARM64)
#    include "emitters/snippets/aarch64/cpu_generator.hpp"
#    include "executors/aarch64/subgraph.hpp"
#    include "transformations/snippets/aarch64/shape_inference.hpp"
#else
#    include "emitters/snippets/x64/cpu_generator.hpp"
#    include "executors/x64/subgraph.hpp"
#    include "transformations/snippets/x64/pass/brgemm_to_brgemm_cpu.hpp"
#    include "transformations/snippets/x64/pass/eliminate_brgemm_copy_b.hpp"
#    include "transformations/snippets/x64/pass/enforce_precision.hpp"
#    include "transformations/snippets/x64/pass/lowered/adjust_brgemm_copy_b_loop_ports.hpp"
#    include "transformations/snippets/x64/pass/lowered/brgemm_cpu_blocking.hpp"
#    include "transformations/snippets/x64/pass/lowered/fuse_load_store_and_convert.hpp"
#    include "transformations/snippets/x64/pass/lowered/insert_brgemm_copy_buffers.hpp"
#    include "transformations/snippets/x64/pass/remove_converts.hpp"
#    include "transformations/snippets/x64/shape_inference.hpp"
#endif

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

#include "utils/cpu_utils.hpp"
#include "utils/ngraph_utils.hpp"

#ifdef SNIPPETS_LIBXSMM_TPP
#    include "snippets/lowered/pass/optimize_domain.hpp"
#    include "transformations/tpp/common/pass/brgemm_to_brgemm_tpp.hpp"
#    include "transformations/tpp/common/pass/lowered/brgemm_tpp_blocking.hpp"
#    include "transformations/tpp/common/pass/lowered/set_tpp_leading_dim.hpp"
#    if defined(OPENVINO_ARCH_X86_64)
#        include "transformations/tpp/x64/pass/eltwise_to_eltwise_tpp.hpp"
#        include "transformations/tpp/x64/pass/fuse_tpp_to_equations.hpp"
#        include "transformations/tpp/x64/pass/scalar_to_scalar_tpp.hpp"
#    endif
#endif

namespace ov::intel_cpu::node {
namespace {

#if defined(OPENVINO_ARCH_X86_64) || defined(OPENVINO_ARCH_ARM64)
struct SubgraphKey {
    SubgraphKey() = default;
    SubgraphKey(std::shared_ptr<SubgraphAttrs> attrs_, std::vector<VectorDims> in_shapes_)
        : attrs(std::move(attrs_)),
          in_shapes(std::move(in_shapes_)) {}
    virtual ~SubgraphKey() = default;

    [[nodiscard]] size_t hash() const {
        using namespace dnnl::impl;
        using namespace dnnl::impl::primitive_hashing;

        size_t seed = get_attr_hash(0, attrs);
        for (const auto& shape : in_shapes) {
            seed = get_vector_hash(seed, shape);
        }

        return seed;
    }
    bool operator==(const SubgraphKey& rhs) const {
        return *attrs == *rhs.attrs && in_shapes == rhs.in_shapes;
    }

    std::shared_ptr<SubgraphAttrs> attrs = nullptr;
    std::vector<VectorDims> in_shapes = {};
};

struct SubgraphCodeGeneratorKey {
    SubgraphCodeGeneratorKey(std::shared_ptr<SubgraphAttrs> attrs_, uint8_t mask_)
        : attrs(std::move(attrs_)),
          broadcasting_mask(mask_) {}

    [[nodiscard]] size_t hash() const {
        using namespace dnnl::impl;
        using namespace dnnl::impl::primitive_hashing;

        size_t seed = get_attr_hash(0, attrs);
        return hash_combine(seed, broadcasting_mask);
    }
    bool operator==(const SubgraphCodeGeneratorKey& rhs) const {
        return *attrs == *rhs.attrs && broadcasting_mask == rhs.broadcasting_mask;
    }

    std::shared_ptr<SubgraphAttrs> attrs = nullptr;
    uint32_t broadcasting_mask = 0;
};
#endif

struct SubgraphShapeInferResultKey {
    SubgraphShapeInferResultKey(std::vector<VectorDims> in_shapes_, uint64_t body_hash_)
        : in_shapes(std::move(in_shapes_)),
          body_hash(body_hash_) {}

    [[nodiscard]] size_t hash() const {
        using namespace dnnl::impl;
        using namespace dnnl::impl::primitive_hashing;

        size_t seed = hash_combine(0, body_hash);
        for (const auto& shape : in_shapes) {
            seed = get_vector_hash(seed, shape);
        }

        return seed;
    }
    bool operator==(const SubgraphShapeInferResultKey& rhs) const {
        return body_hash == rhs.body_hash && in_shapes == rhs.in_shapes;
    }

    std::vector<VectorDims> in_shapes = {};
    uint64_t body_hash = 0;
};

struct SubgraphShapeInferResult {
    SubgraphShapeInferResult(IShapeInfer::Result res) : result(std::move(res)) {}

    IShapeInfer::Result result;
};

}  // namespace

static _ov_dnnl_cpu_isa getHostIsa() {
#if defined(OPENVINO_ARCH_ARM64)
    return dnnl::impl::cpu::aarch64::asimd;
#else
    return dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ? dnnl::impl::cpu::x64::avx512_core
                                                                            : dnnl::impl::cpu::x64::avx2;
#endif
}

Subgraph::Subgraph(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, SnippetShapeInferFactory(op)),
      host_isa(getHostIsa()),
      subgraph_attrs(std::make_shared<SubgraphAttrs>()) {
    const auto& tmp_snippet = ov::as_type_ptr<snippets::op::Subgraph>(op);
    OPENVINO_ASSERT(tmp_snippet, "Attempt to create Subgraph node from an invalid op type");
    subgraph_attrs->snippet = tmp_snippet->clone();
    subgraph_attrs->bodyHash = getBodyHash(tmp_snippet);

#if defined(OPENVINO_ARCH_ARM64)
    subgraph_attrs->snippet->set_generator(
        std::make_shared<aarch64::CPUGenerator>(host_isa, context->getParamsCache()));
#elif defined(OPENVINO_ARCH_X86_64)
    subgraph_attrs->snippet->set_generator(std::make_shared<CPUGenerator>(host_isa, context->getParamsCache()));
#else
    THROW_CPU_NODE_ERR("Subgraphs code-generator is not supported on non-x64 platforms");
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
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    const std::set<ov::element::Type> supportedPrecisions =
        {ov::element::f32, ov::element::i32, ov::element::bf16, ov::element::f16, ov::element::i8, ov::element::u8};

    bool dimRanksAreEqual = true;
    for (size_t i = 0; dimRanksAreEqual && i < inputShapes.size(); i++) {
        for (size_t j = 0; dimRanksAreEqual && j < outputShapes.size(); j++) {
            if (inputShapes[i].getRank() != outputShapes[j].getRank()) {
                dimRanksAreEqual = false;
            }
        }
    }

    const size_t ndims = outputShapes[0].getRank();
    // Domain sensitive operations and dynamic Subgraphs support only Planar layout
    const bool isOnlyPlanarApplicable = subgraph_attrs->snippet->has_domain_sensitive_ops();
    const bool isChannelsFirstApplicable = dnnl::impl::utils::one_of(ndims, 1u, 2u, 3u, 4u, 5u) && dimRanksAreEqual &&
                                           !isOnlyPlanarApplicable && !isDynamic;
    // Todo: Subgraphs currently don't support per-channel broadcasting of Blocked descriptors because
    //  canonicalization can't distinguish between <N, C, H, W, c> and <N, C, D, H, W> cases.
    //  See snippets::op::Subgraph::canonicalize for details.
#if defined(OPENVINO_ARCH_ARM64)
    bool isBlockedApplicable = false;
#else
    bool isBlockedApplicable =
        dnnl::impl::utils::one_of(ndims, 3u, 4u, 5u) && dimRanksAreEqual && !isOnlyPlanarApplicable && !isDynamic;

    for (const auto& inShape : inputShapes) {
        if (isDynamic && inShape.getRank() != 1) {
            isBlockedApplicable =
                isBlockedApplicable && inShape.getMinDims()[1] != Shape::UNDEFINED_DIM && inShape.getMinDims()[1] > 1;
        }
    }
#endif

    enum LayoutType : uint8_t { Planar, ChannelsFirst, Blocked };
    auto initDesc = [&](LayoutType lt) -> NodeDesc {
        auto createMemoryDesc =
            [lt](const Shape& shape, ov::element::Type prc, size_t offset) -> std::shared_ptr<CpuBlockedMemoryDesc> {
            const auto& dims = shape.getDims();
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
            }
            if (lt == Blocked && shape.getRank() != 1 &&
                (shape.getMinDims()[1] != Shape::UNDEFINED_DIM && shape.getMinDims()[1] > 1)) {
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
            }
            VectorDims blocks = dims;
            VectorDims order(blocks.size());
            std::iota(order.begin(), order.end(), 0);

            return std::make_shared<CpuBlockedMemoryDesc>(prc, shape, blocks, order, offset);
        };

        size_t offset = 0;
        NodeConfig config;
        config.inConfs.resize(inputShapes.size());
        for (size_t i = 0; i < inputShapes.size(); i++) {
            const auto originalInputPrecision = getOriginalInputPrecisionAtPort(i);
            const auto precision =
                ((originalInputPrecision == ov::element::f32) &&
                 one_of(context->getConfig().inferencePrecision, ov::element::bf16, ov::element::f16) &&
                 subgraph_attrs->snippet->has_domain_sensitive_ops())
                    ? context->getConfig().inferencePrecision
                    : originalInputPrecision;
            if (supportedPrecisions.count(precision) == 0) {
                THROW_CPU_NODE_ERR("doesn't support ", precision, " precision.");
            }

            const auto equalPrecisions =
                getOriginalOutputPrecisions().size() == 1 && precision == getOriginalOutputPrecisionAtPort(0);

            BlockedMemoryDesc::CmpMask inputMask = BlockedMemoryDesc::SKIP_OFFSET_MASK;
            PortConfig portConfig;
            portConfig.inPlace((!i && canBeInPlace() && equalPrecisions) ? 0 : -1);
            portConfig.constant(false);
            if (inputShapes[i].getDims()[0] == 1) {
                inputMask.reset(0);  // accepts any stride on batch axis
            }
            portConfig.setMemDesc(createMemoryDesc(inputShapes[i], precision, offset), inputMask);
            config.inConfs[i] = portConfig;
        }
        config.outConfs.resize(outputShapes.size());
        for (size_t i = 0; i < outputShapes.size(); i++) {
            auto precision = getOriginalOutputPrecisionAtPort(i);
            if (supportedPrecisions.count(precision) == 0) {
                THROW_CPU_NODE_ERR("doesn't support ", precision, " precision.");
            }

            BlockedMemoryDesc::CmpMask outputMask = BlockedMemoryDesc::SKIP_OFFSET_MASK;
            PortConfig portConfig;
            portConfig.inPlace(-1);
            portConfig.constant(false);
            if (outputShapes[i].getDims()[0] == 1) {
                outputMask.reset(0);  // accepts any stride on batch axis
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

    if (isChannelsFirstApplicable) {
        supportedPrimitiveDescriptors.emplace_back(initDesc(ChannelsFirst));
    }
    if (isBlockedApplicable) {
        supportedPrimitiveDescriptors.emplace_back(initDesc(Blocked));
    }
    supportedPrimitiveDescriptors.emplace_back(initDesc(Planar));
}

void Subgraph::selectOptimalPrimitiveDescriptor() {
    selectPreferPrimitiveDescriptorWithShape(getImplPriority(), true);
}

ov::element::Type Subgraph::getRuntimePrecision() const {
    std::vector<ov::element::Type> inputPrecisions;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge && parentEdge->getStatus() == Edge::Status::Validated &&
            !parentEdge->getParent()->isConstant()) {
            inputPrecisions.emplace_back(
                DnnlExtensionUtils::DataTypeToElementType((parentEdge->getMemoryPtr()->getDataType())));
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
    for (size_t i = 0; i < input_num; i++) {
        srcMemPtrs[i] = getSrcMemoryAtPort(i);
    }
    for (size_t i = 0; i < output_num; i++) {
        dstMemPtrs[i] = getDstMemoryAtPort(i);
    }
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
    for (size_t i = 0; i < input_num; i++) {
        start_offset_in[i] = get_offset(srcMemPtrs[i]->getDescWithType<BlockedMemoryDesc>());
    }
    for (size_t i = 0; i < output_num; i++) {
        start_offset_out[i] = get_offset(dstMemPtrs[i]->getDescWithType<BlockedMemoryDesc>());
    }
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
    for (const auto& p : subgraph_attrs->inMemPrecs) {
        precisions.first.push_back(p);
    }
    for (const auto& p : subgraph_attrs->outMemPrecs) {
        precisions.second.push_back(p);
    }
    return precisions;
}

void Subgraph::initPluginBlockedShapes() const {
    in_shapes.resize(input_num);
    for (size_t i = 0; i < srcMemPtrs.size(); i++) {
        in_shapes[i] = srcMemPtrs[i]->getDescWithType<BlockedMemoryDesc>()->getBlockDims();
    }
}

Subgraph::DataFlowPasses Subgraph::getDataFlowPasses() {
    DataFlowPasses backend_passes;

    using PassPosition = ov::snippets::pass::PassPosition;
    using Place = PassPosition::Place;

#define SNIPPETS_REGISTER_PASS_ABSOLUTE_COMMON(PASS_PLACE, PASS, ...) \
    backend_passes.emplace_back(PassPosition(PASS_PLACE), std::make_shared<PASS>(__VA_ARGS__))
#define SNIPPETS_REGISTER_PASS_RELATIVE_COMMON(PASS_PLACE, TARGET_PASS, PASS, ...)             \
    backend_passes.emplace_back(PassPosition(PASS_PLACE, TARGET_PASS::get_type_info_static()), \
                                std::make_shared<PASS>(__VA_ARGS__))

#if defined(OPENVINO_ARCH_X86_64)
#    define SNIPPETS_REGISTER_PASS_ABSOLUTE_X86_64(PASS_PLACE, PASS, ...) \
        backend_passes.emplace_back(PassPosition(PASS_PLACE), std::make_shared<PASS>(__VA_ARGS__))
#    define SNIPPETS_REGISTER_PASS_RELATIVE_X86_64(PASS_PLACE, TARGET_PASS, PASS, ...)             \
        backend_passes.emplace_back(PassPosition(PASS_PLACE, TARGET_PASS::get_type_info_static()), \
                                    std::make_shared<PASS>(__VA_ARGS__))
#else
#    define SNIPPETS_REGISTER_PASS_ABSOLUTE_X86_64(PASS_PLACE, PASS, ...)
#    define SNIPPETS_REGISTER_PASS_RELATIVE_X86_64(PASS_PLACE, TARGET_PASS, PASS, ...)
#endif  // OPENVINO_ARCH_X86_64

#if defined(OPENVINO_ARCH_ARM64)
#    define SNIPPETS_REGISTER_PASS_RELATIVE_ARM64(PASS_PLACE, TARGET_PASS, PASS, ...)              \
        backend_passes.emplace_back(PassPosition(PASS_PLACE, TARGET_PASS::get_type_info_static()), \
                                    std::make_shared<PASS>(__VA_ARGS__))
#else
#    define SNIPPETS_REGISTER_PASS_RELATIVE_ARM64(PASS_PLACE, TARGET_PASS, PASS, ...)
#endif  // OPENVINO_ARCH_ARM64

    SNIPPETS_REGISTER_PASS_ABSOLUTE_COMMON(Place::PipelineStart, ConvertToSwishCPU);
    SNIPPETS_REGISTER_PASS_RELATIVE_COMMON(Place::After,
                                           ov::snippets::pass::Canonicalization,
                                           ov::snippets::pass::AnalyzeBroadcastableInputs,
                                           broadcastable_inputs);

    if (one_of(context->getConfig().inferencePrecision, ov::element::bf16, ov::element::f16) &&
        subgraph_attrs->snippet->has_domain_sensitive_ops()) {
        // enforce BF16 precisions to supported operations
        // MatMul has to be decomposed to Brgemm operations before enforcement
        // Note, MatMul decomposition will be run later again for case if BF16 enforcement is not happened
        SNIPPETS_REGISTER_PASS_ABSOLUTE_X86_64(Place::PipelineStart, ov::snippets::pass::MatMulToBrgemm);
        SNIPPETS_REGISTER_PASS_RELATIVE_X86_64(Place::After,
                                               ov::snippets::pass::MatMulToBrgemm,
                                               pass::EnforcePrecision,
                                               element::f32,
                                               context->getConfig().inferencePrecision);
    }
    SNIPPETS_REGISTER_PASS_RELATIVE_X86_64(Place::Before,
                                           ov::snippets::pass::PropagatePrecision,
                                           ov::intel_cpu::pass::BrgemmToBrgemmCPU);
    SNIPPETS_REGISTER_PASS_RELATIVE_X86_64(Place::After,
                                           ov::intel_cpu::pass::BrgemmToBrgemmCPU,
                                           ov::intel_cpu::pass::EliminateBrgemmCopyB);
    SNIPPETS_REGISTER_PASS_ABSOLUTE_X86_64(Place::PipelineEnd, ov::intel_cpu::pass::RemoveConverts);
    SNIPPETS_REGISTER_PASS_ABSOLUTE_COMMON(Place::PipelineEnd, ov::intel_cpu::pass::MulAddToFMA);

#ifdef SNIPPETS_LIBXSMM_TPP
    SNIPPETS_REGISTER_PASS_RELATIVE_X86_64(Place::Before,
                                           ov::intel_cpu::pass::BrgemmToBrgemmCPU,
                                           ov::intel_cpu::tpp::pass::BrgemmToBrgemmTPP);
    // Note: There could be several ConvertConstantsToScalars instances in the pipeline
    SNIPPETS_REGISTER_PASS_ABSOLUTE_X86_64(Place::PipelineEnd, ov::intel_cpu::tpp::pass::ScalarToScalarTPP);
    SNIPPETS_REGISTER_PASS_RELATIVE_X86_64(Place::After,
                                           ov::intel_cpu::tpp::pass::BrgemmToBrgemmTPP,
                                           ov::intel_cpu::tpp::pass::EltwiseToEltwiseTPP);
    SNIPPETS_REGISTER_PASS_RELATIVE_X86_64(Place::After,
                                           ov::intel_cpu::tpp::pass::EltwiseToEltwiseTPP,
                                           ov::intel_cpu::tpp::pass::FuseTPPToEquations);
    SNIPPETS_REGISTER_PASS_RELATIVE_ARM64(Place::Before,
                                          ov::snippets::pass::PropagatePrecision,
                                          ov::intel_cpu::tpp::pass::BrgemmToBrgemmTPP);
#endif

#undef SNIPPETS_REGISTER_PASS_ABSOLUTE_COMMON
#undef SNIPPETS_REGISTER_PASS_RELATIVE_COMMON
#undef SNIPPETS_REGISTER_PASS_ABSOLUTE_X86_64
#undef SNIPPETS_REGISTER_PASS_RELATIVE_X86_64
#undef SNIPPETS_REGISTER_PASS_RELATIVE_ARM64

    return backend_passes;
}

Subgraph::ControlFlowPasses Subgraph::getControlFlowPasses() const {
    ControlFlowPasses backend_passes;
#if defined(OPENVINO_ARCH_X86_64) || (defined(OPENVINO_ARCH_ARM64) && defined(SNIPPETS_LIBXSMM_TPP))
    using PassPosition = ov::snippets::pass::PassPosition;
    using Place = PassPosition::Place;
#endif

#if defined(OPENVINO_ARCH_X86_64)
#    define SNIPPETS_REGISTER_PASS_RELATIVE_X86_64(PASS_PLACE, TARGET_PASS, PASS, ...)             \
        backend_passes.emplace_back(PassPosition(PASS_PLACE, TARGET_PASS::get_type_info_static()), \
                                    std::make_shared<PASS>(__VA_ARGS__))
#else
#    define SNIPPETS_REGISTER_PASS_RELATIVE_X86_64(PASS_PLACE, TARGET_PASS, PASS, ...)
#endif  // OPENVINO_ARCH_X86_64

#if defined(OPENVINO_ARCH_ARM64)
#    define SNIPPETS_REGISTER_PASS_RELATIVE_ARM64(PASS_PLACE, TARGET_PASS, PASS, ...)              \
        backend_passes.emplace_back(PassPosition(PASS_PLACE, TARGET_PASS::get_type_info_static()), \
                                    std::make_shared<PASS>(__VA_ARGS__))
#else
#    define SNIPPETS_REGISTER_PASS_RELATIVE_ARM64(PASS_PLACE, TARGET_PASS, PASS, ...)
#endif  // OPENVINO_ARCH_ARM64

    SNIPPETS_REGISTER_PASS_RELATIVE_X86_64(Place::After,
                                           ov::snippets::lowered::pass::MarkLoops,
                                           ov::intel_cpu::pass::BrgemmCPUBlocking);
#ifdef SNIPPETS_DEBUG_CAPS
    const auto& debug_config = subgraph_attrs->snippet->get_debug_config();
    if (debug_config.perf_count_mode != snippets::DebugCapsConfig::PerfCountMode::Disabled) {
        SNIPPETS_REGISTER_PASS_RELATIVE_X86_64(Place::After,
                                               ov::intel_cpu::pass::BrgemmCPUBlocking,
                                               ov::snippets::lowered::pass::InsertPerfCountVerbose,
                                               getName());
    }
#endif  // SNIPPETS_DEBUG_CAPS

    SNIPPETS_REGISTER_PASS_RELATIVE_X86_64(Place::After,
                                           ov::snippets::lowered::pass::InitLoops,
                                           ov::intel_cpu::pass::AdjustBrgemmCopyBLoopPorts);

    SNIPPETS_REGISTER_PASS_RELATIVE_X86_64(Place::After,
                                           ov::snippets::lowered::pass::InsertLoops,
                                           ov::intel_cpu::pass::FuseLoadStoreConvert);
    SNIPPETS_REGISTER_PASS_RELATIVE_X86_64(Place::Before,
                                           ov::snippets::lowered::pass::InsertBuffers,
                                           ov::intel_cpu::pass::InsertBrgemmCopyBuffers);

#ifdef SNIPPETS_LIBXSMM_TPP
    SNIPPETS_REGISTER_PASS_RELATIVE_X86_64(Place::Before,
                                           ov::intel_cpu::pass::BrgemmCPUBlocking,
                                           ov::intel_cpu::tpp::pass::BrgemmTPPBlocking);
    SNIPPETS_REGISTER_PASS_RELATIVE_X86_64(Place::After,
                                           ov::intel_cpu::pass::FuseLoadStoreConvert,
                                           ov::intel_cpu::tpp::pass::SetTPPLeadingDim);
    SNIPPETS_REGISTER_PASS_RELATIVE_ARM64(Place::After,
                                          ov::snippets::lowered::pass::MarkLoops,
                                          ov::intel_cpu::tpp::pass::BrgemmTPPBlocking);
    SNIPPETS_REGISTER_PASS_RELATIVE_ARM64(Place::After,
                                          ov::snippets::lowered::pass::InsertLoops,
                                          ov::intel_cpu::tpp::pass::SetTPPLeadingDim);
#endif

#undef SNIPPETS_REGISTER_PASS_RELATIVE_X86_64
#undef SNIPPETS_REGISTER_PASS_RELATIVE_ARM64
    return backend_passes;
}

uint32_t Subgraph::getBroadcastingMask(const std::vector<VectorDims>& input_shapes) {
    uint32_t mask = 0;
    OPENVINO_ASSERT(broadcastable_inputs.size() <= sizeof(mask) * CHAR_BIT,
                    "Incorrect size of broadcastable inputs of Subgraph");
    for (const auto& broadcastable_input : broadcastable_inputs) {
        const auto& shape = input_shapes[broadcastable_input.first];
        mask = mask << 1;
        if (*(shape.rbegin() + broadcastable_input.second) == 1) {
            mask = mask | 1;
        }
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
    for (const auto& s : in_blocked_shapes) {
        in_shapes.emplace_back(s.first);
    }
    subgraph->shape_infer(in_shapes);

    const auto control_flow_config = std::make_shared<ov::snippets::lowered::pass::PassConfig>();
    const auto control_flow_passes = getControlFlowPasses();

#ifdef SNIPPETS_LIBXSMM_TPP
    // Note: temporary disabled. Re-enable after ticket 132833 is resolved
    control_flow_config->disable<ov::snippets::lowered::pass::OptimizeDomain>();

    subgraph->set_tile_rank(std::min(2ul, subgraph->infer_master_shape().size()));
#endif

    // Note: minimal JIT work amount is a predefined value that describes the number of kernel iterations (work amount)
    // needed to cover kernel call overhead. It is used for balancing between parallel and JIT work amounts in domain
    // optimization.
    subgraph->control_flow_transformations(static_cast<size_t>(parallel_get_max_threads()),
                                           256,
                                           std::make_shared<snippets::CPUShapeInferSnippetsFactory>(),
                                           control_flow_config,
                                           control_flow_passes);
}

void Subgraph::prepareParams() {
#if defined(OPENVINO_ARCH_X86_64) || defined(OPENVINO_ARCH_ARM64)
    const auto& cache = context->getParamsCache();

    auto builder = [this, &cache](const SubgraphKey& key) -> std::shared_ptr<SubgraphBaseExecutor> {
        const auto& snippet = subgraph_attrs->snippet;

        SubgraphBaseExecutor::BufferScratchpadAllocator allocator = [this](size_t size) {
            return getScratchPadMem(std::make_shared<CpuBlockedMemoryDesc>(ov::element::u8, intel_cpu::Shape{size}));
        };

        if (is_dynamic) {
            // Dynamic case:
            // 1. Generate JIT code if needed
            // 2. Update runtime config with dynamic values
            //    If JIT code has been taken from cache, need to set cached kernel executor table for the configuration
            // 3. Create SubgraphDynamicSpecializedExecutor
            const auto code_gen_result = cache->getOrCreate(
                SubgraphCodeGeneratorKey(subgraph_attrs, getBroadcastingMask(in_shapes)),
                [](const SubgraphCodeGeneratorKey& key) -> std::shared_ptr<SubgraphCodeGenerator> {
                    return std::make_shared<SubgraphCodeGenerator>(key.attrs, std::make_shared<CPURuntimeConfig>());
                });
            const auto& code_gen = code_gen_result.first;
            // [148644] : Update Kernel table from SubgraphCodeGenerator when JIT code was already generated with
            // specific Kernel table
            if (code_gen_result.second == CacheEntryBase::LookUpStatus::Hit) {
                snippet->get_runtime_configurator()->set_kernel_executor_table(
                    code_gen->get()->lowering_result.kernel_executor_table);
            }
            const auto& snippet_config = ov::as_type_ptr<CPURuntimeConfig>(snippet->update_runtime_config());
            return std::make_shared<SubgraphDynamicSpecializedExecutor>(snippet_config,
                                                                        key.attrs,
                                                                        code_gen,
                                                                        start_offset_in,
                                                                        start_offset_out,
                                                                        allocator,
                                                                        cache);
        }  // Static case:
        // 1. Update runtime config to get static scheduling data (io data offsets, parallel domain) which will be
        // compiled in JIT code
        // 2. Generate JIT code with this static data if needed
        // 3. Create SubgraphStaticExecutor
        const auto& snippet_config = ov::as_type_ptr<CPURuntimeConfig>(snippet->update_runtime_config());
        const auto code_gen_result = cache->getOrCreate(
            SubgraphCodeGeneratorKey(subgraph_attrs, getBroadcastingMask(in_shapes)),
            [&snippet_config](const SubgraphCodeGeneratorKey& key) -> std::shared_ptr<SubgraphCodeGenerator> {
                return std::make_shared<SubgraphCodeGenerator>(key.attrs, snippet_config);
            });
        return std::make_shared<SubgraphStaticExecutor>(snippet_config,
                                                        key.attrs,
                                                        code_gen_result.first,
                                                        start_offset_in,
                                                        start_offset_out,
                                                        allocator,
                                                        cache);
    };

    const auto result = cache->getOrCreate(SubgraphKey(subgraph_attrs, in_shapes), builder);
    execPtr = result.first;
#endif

    OPENVINO_ASSERT(execPtr != nullptr, "Executor is not created for node ", getName(), ".");
}

IShapeInfer::Result Subgraph::shapeInfer() const {
    for (size_t i = 0; i < srcMemPtrs.size(); i++) {
        in_shapes[i] = srcMemPtrs[i]->getDescWithType<BlockedMemoryDesc>()->getBlockDims();
    }

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
        if (parent->getChildEdges().size() != 1) {
            return false;
        }

        // WA to prevent memory corruption caused by inplace feature
        if (parent->getType() == Type::Concatenation) {
            for (auto& parentParentEdge : parent->getParentEdges()) {
                auto parentParent = parentParentEdge.lock()->getParent();
                if (parentParent->getChildEdges().size() != 1) {
                    return false;
                }
            }
        }
    }
    return getInputShapeAtPort(0) == getOutputShapeAtPort(0);
}

bool Subgraph::created() const {
    return getType() == Type::Subgraph;
}

void Subgraph::execute(const dnnl::stream& strm) {
    OPENVINO_ASSERT(execPtr, "Can't execute Subgraph node. Primitive didn't created");
    execPtr->execute(strm, srcMemPtrs, dstMemPtrs);
}

void Subgraph::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

}  // namespace ov::intel_cpu::node
