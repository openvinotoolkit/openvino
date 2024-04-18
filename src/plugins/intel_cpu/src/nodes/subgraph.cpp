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

namespace ov {
namespace intel_cpu {
namespace node {
namespace {

struct SubgraphKey {
protected:
    SubgraphKey() = default;
    SubgraphKey(const std::shared_ptr<Subgraph::SubgraphAttrs>& attrs_) : attrs(attrs_) {}
    virtual ~SubgraphKey() = default;

    size_t hash() const;

public:
    std::shared_ptr<Subgraph::SubgraphAttrs> attrs = nullptr;
};

struct SubgraphSpecializedKey : public SubgraphKey {
    SubgraphSpecializedKey(const std::shared_ptr<Subgraph::SubgraphAttrs>& attrs_, const std::vector<VectorDims>& in_shapes_)
        : SubgraphKey(attrs_), in_shapes(in_shapes_) {}

    size_t hash() const;
    bool operator==(const SubgraphSpecializedKey& rhs) const;

    std::vector<VectorDims> in_shapes = {};
};

struct SubgraphShapeAgnosticKey : public SubgraphKey {
    SubgraphShapeAgnosticKey(const std::shared_ptr<Subgraph::SubgraphAttrs>& attrs_, uint8_t mask_)
        : SubgraphKey(attrs_), mask(mask_) {}

    size_t hash() const;
    bool operator==(const SubgraphShapeAgnosticKey& rhs) const;

    uint8_t mask = 0;
};

struct SubgraphShapeInferResultKey {
    SubgraphShapeInferResultKey(std::vector<VectorDims> in_shapes_, uint64_t body_hash_)
        : in_shapes(std::move(in_shapes_)), body_hash(body_hash_) {}

    size_t hash() const;
    bool operator==(const SubgraphShapeInferResultKey& rhs) const;

    std::vector<VectorDims> in_shapes = {};
    uint64_t body_hash = 0;
};

size_t SubgraphKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
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

size_t SubgraphSpecializedKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = SubgraphKey::hash();
    for (const auto& shape : in_shapes)
        seed = get_vector_hash(seed, shape);

    return seed;
}

size_t SubgraphShapeAgnosticKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = SubgraphKey::hash();
    seed = hash_combine(seed, mask);

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

bool operator!=(const Subgraph::SubgraphAttrs& lhs, const Subgraph::SubgraphAttrs& rhs) {
    return !(lhs == rhs);
}

bool SubgraphSpecializedKey::operator==(const SubgraphSpecializedKey& rhs) const {
    if (*attrs != *rhs.attrs)
        return false;
    if (in_shapes != rhs.in_shapes)
        return false;
    return true;
}

bool SubgraphShapeAgnosticKey::operator==(const SubgraphShapeAgnosticKey& rhs) const {
    if (*attrs != *rhs.attrs)
        return false;
    if (mask != rhs.mask)
        return false;
    return true;
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
        : Node(op, context, SnippetShapeInferFactory(op)), snippetAttrs(std::make_shared<SubgraphAttrs>()) {
    host_isa = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core) ?
        dnnl::impl::cpu::x64::avx512_core : dnnl::impl::cpu::x64::avx2;
    const auto& tmp_snippet = ov::as_type_ptr<snippets::op::Subgraph>(op);
    OPENVINO_ASSERT(tmp_snippet, "Attempt to create Subgraph node from an invalid op type");
    snippetAttrs->snippet = tmp_snippet->clone();
    snippetAttrs->bodyHash = get_body_hash(tmp_snippet);

#if defined(OPENVINO_ARCH_X86_64)
    snippetAttrs->snippet->set_generator(std::make_shared<CPUGenerator>(host_isa, context->getParamsCache()));
#else
    OPENVINO_THROW("CPU plugin: Subgraphs code-generator is not supported on non-x64 platforms");

#endif // OPENVINO_ARCH_X86_64

    // Note: we have to update shapeInfer, so it uses the per-thread op::Subgraph copy
    shapeInference = SnippetShapeInferFactory(snippetAttrs->snippet).makeShapeInfer();
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
    // Domain sensitive operations and dynamic Subgraphs support only Planar layout
    const bool isOnlyPlanarApplicable = snippetAttrs->snippet->has_domain_sensitive_ops();
    const bool isChannelsFirstApplicable = dnnl::impl::utils::one_of(ndims, 1u, 2u, 3u, 4u, 5u) && dimRanksAreEqual && !isOnlyPlanarApplicable && !isDynamic;
    // Todo: Subgraphs currently don't support per-channel broadcasting of Blocked descriptors because
    //  canonicalization can't distinguish between <N, C, H, W, c> and <N, C, D, H, W> cases.
    //  See snippets::op::Subgraph::canonicalize for details.
    bool isBlockedApplicable = dnnl::impl::utils::one_of(ndims,  3u, 4u, 5u) && dimRanksAreEqual && !isOnlyPlanarApplicable && !isDynamic;

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
                                     snippetAttrs->snippet->has_domain_sensitive_ops()) ?
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

        in_shapes.resize(input_num);

        init_memory_ptrs();
        init_attrs();
        init_start_offsets();
        lower();
    }

    Node::createPrimitive();
}

void Subgraph::init_memory_ptrs() {
    srcMemPtrs.resize(input_num);
    dstMemPtrs.resize(output_num);
    for (size_t i = 0; i < input_num; i++)
        srcMemPtrs[i] = getSrcMemoryAtPort(i);
    for (size_t i = 0; i < output_num; i++)
        dstMemPtrs[i] = getDstMemoryAtPort(i);
}

void Subgraph::init_attrs() {
    const auto config = getSelectedPrimitiveDescriptor()->getConfig();

    snippetAttrs->inMemPrecs.resize(input_num);
    snippetAttrs->outMemPrecs.resize(output_num);

    snippetAttrs->inMemOrders.resize(input_num);
    snippetAttrs->outMemOrders.resize(output_num);

    for (size_t i = 0; i < input_num; i++) {
        const auto& memDesc = srcMemPtrs[i]->getDescWithType<BlockedMemoryDesc>();
        snippetAttrs->inMemPrecs[i] = memDesc->getPrecision();
        snippetAttrs->inMemOrders[i] = memDesc->getOrder();
    }
    for (size_t i = 0; i < output_num; i++) {
        const auto& memDesc = dstMemPtrs[i]->getDescWithType<BlockedMemoryDesc>();
        snippetAttrs->outMemPrecs[i] = memDesc->getPrecision();
        snippetAttrs->outMemOrders[i] = memDesc->getOrder();
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

void Subgraph::init_snippets_blocked_shapes(snippets::op::Subgraph::BlockedShapeVector& in_blocked_shapes) const {
    const auto& config = getSelectedPrimitiveDescriptor()->getConfig();

    in_blocked_shapes.reserve(input_num);
    for (size_t i = 0; i < input_num; i++) {
        const auto& memDesc = config.inConfs[i].getMemDesc();
        const auto& blockedDesc = memDesc->as<BlockedMemoryDesc>();
        const auto& order = blockedDesc->getOrder();

        in_blocked_shapes.emplace_back(blockedDesc->getBlockDims(), order);
    }
}

void Subgraph::init_precisions(std::vector<ov::element::Type>& input_types, std::vector<ov::element::Type>& output_types) const {
    input_types.reserve(input_num);
    output_types.reserve(output_num);
    for (const auto& p : snippetAttrs->inMemPrecs)
        input_types.push_back(p);
    for (const auto& p : snippetAttrs->outMemPrecs)
        output_types.push_back(p);
}

Subgraph::DataFlowPasses Subgraph::get_data_flow_passes() const {
    DataFlowPasses backend_passes;

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
    if (context->getConfig().inferencePrecision == ov::element::bf16 && snippetAttrs->snippet->has_domain_sensitive_ops()) {
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

#undef SNIPPETS_REGISTER_PASS_ABSOLUTE
#undef SNIPPETS_REGISTER_PASS_RELATIVE

    return backend_passes;
}

std::pair<Subgraph::ControlFlowConfig, Subgraph::ControlFlowPasses> Subgraph::get_control_flow_passes() const {
    ControlFlowConfig lowering_config = std::make_shared<ov::snippets::lowered::pass::PassConfig>();
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
                                    ov::intel_cpu::pass::BrgemmBlocking);
    SNIPPETS_REGISTER_PASS_RELATIVE(Place::After, ov::snippets::lowered::pass::InsertLoops,
                                    ov::intel_cpu::pass::FuseLoadStoreConvert);
    SNIPPETS_REGISTER_PASS_RELATIVE(Place::After, ov::intel_cpu::pass::FuseLoadStoreConvert,
                                    ov::intel_cpu::pass::SetBrgemmCopyBBuffersShape);

#ifdef SNIPPETS_LIBXSMM_TPP
    // Note: temporary disabled. Re-enable after ticket 132833 is resolved
    lowering_config->disable<ov::snippets::lowered::pass::OptimizeDomain>();
    SNIPPETS_REGISTER_PASS_RELATIVE(Place::After, ov::intel_cpu::pass::FuseLoadStoreConvert,
                                    ov::intel_cpu::tpp::pass::SetTPPLeadingDim);
#endif

#undef SNIPPETS_REGISTER_PASS_RELATIVE
    return std::make_pair(lowering_config, backend_passes);
}

uint8_t Subgraph::get_broadcasting_mask(const std::vector<VectorDims>& input_shapes) const {
    uint8_t mask = 0;
    // TODO: add check for non-eltwise inputs
    for (const auto& shape : input_shapes) {
        mask = mask << 1;
        if (shape.back() == 1)
            mask = mask | 1;
    }
    return mask;
}

bool Subgraph::need_blocked_shape_infer() const {
    const auto& inConfs = getSelectedPrimitiveDescriptor()->getConfig().inConfs;
    return std::any_of(inConfs.cbegin(), inConfs.cend(), [](const PortConfig& conf) {
        return !conf.getMemDesc()->as<BlockedMemoryDesc>()->hasLayoutType(LayoutType::ncsp);
    });
}

void Subgraph::lower() {
    snippets::op::Subgraph::BlockedShapeVector in_blocked_shapes;
    std::vector<ov::element::Type> input_precisions, output_precisions;
    init_snippets_blocked_shapes(in_blocked_shapes);
    init_precisions(input_precisions, output_precisions);

    const auto& subgraph = snippetAttrs->snippet;

    subgraph->data_flow_transformations(in_blocked_shapes, input_precisions, output_precisions, get_data_flow_passes());

    // TODO: Snippets don't support backend-provided blocking, so we need to reshape body
    //       using blocked shapes first. This can be removed after [121670]
    if (need_blocked_shape_infer()) {
        std::vector<snippets::VectorDimsRef> in_shapes;
        for (const auto& s : in_blocked_shapes)
            in_shapes.emplace_back(s.first);
        subgraph->shape_infer(in_shapes);
    }

    // Note: minimal JIT work amount is a predefined value that describes the number of kernel iterations (work amount)
    // needed to cover kernel call overhead. It is used for balancing between parallel and JIT work amounts in domain optimization.
#ifdef SNIPPETS_LIBXSMM_TPP
    const auto& lir = subgraph->convert_body_to_linear_ir(static_cast<size_t>(parallel_get_max_threads()), 256,
                                                          std::make_shared<snippets::CPUShapeInferSnippetsFactory>());
    lir->set_loop_depth(std::min(2ul, lir->get_master_shape().size()));
#else
    subgraph->convert_body_to_linear_ir(static_cast<size_t>(parallel_get_max_threads()), 256,
                                        std::make_shared<snippets::CPUShapeInferSnippetsFactory>());
#endif

    const auto control_flow_settings = get_control_flow_passes();
    subgraph->lower(control_flow_settings.first, control_flow_settings.second);
}

void Subgraph::prepareParams() {
    const auto cache = context->getParamsCache();

    auto builder = [this, cache](const SubgraphSpecializedKey& key) -> std::shared_ptr<SubgraphExecutor> {
        if (is_dynamic) {
            auto shape_agnostic_builder = [](const SubgraphShapeAgnosticKey& key) -> std::shared_ptr<SubgraphExecutor> {
                return std::make_shared<SubgraphJitShapeAgnosticExecutor>(key.attrs);
            };

            const auto shape_agnostic_key = SubgraphShapeAgnosticKey(snippetAttrs, get_broadcasting_mask(in_shapes));
            const auto shape_agnostic_result = cache->getOrCreate(shape_agnostic_key, shape_agnostic_builder);
            const auto& shape_agnostic_exec_ptr = std::dynamic_pointer_cast<SubgraphJitShapeAgnosticExecutor>(shape_agnostic_result.first);
            OPENVINO_ASSERT(shape_agnostic_exec_ptr != nullptr, "ShapeAgnosticExecutor is not created for node ", getName(), ".");

            return std::make_shared<SubgraphJitDynamicSpecializedExecutor>(key.attrs, start_offset_in, start_offset_out, shape_agnostic_exec_ptr);
        } else {
            return std::make_shared<SubgraphJitStaticExecutor>(key.attrs, start_offset_in, start_offset_out);
        }
    };

    const auto result = cache->getOrCreate(SubgraphSpecializedKey(snippetAttrs, in_shapes), builder);
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
    const auto result = cache->getOrCreate(SubgraphShapeInferResultKey(in_shapes, snippetAttrs->bodyHash), builder);
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

Subgraph::SubgraphJitExecutor::SubgraphJitExecutor(const std::shared_ptr<SubgraphAttrs>& snippet_attrs)
    : SubgraphExecutor() {
#if defined(__linux__) && defined(SNIPPETS_DEBUG_CAPS)
    const auto target = std::dynamic_pointer_cast<const CPUTargetMachine>(snippet_attrs->snippet->get_generator()->get_target_machine());
    enabled_segfault_detector = target && target->debug_config.enable_segfault_detector;
#endif
}

void Subgraph::SubgraphJitExecutor::generate(const std::shared_ptr<SubgraphAttrs>& snippet_attrs, const std::shared_ptr<CPURuntimeConfig>& cpu_config) {
    jit_snippets_compile_args jcp;
    jcp.master_shape = cpu_config->parallel_domain;
    jcp.data_offsets = cpu_config->io_data_offsets;
    schedule = std::make_shared<ov::snippets::Schedule>(snippet_attrs->snippet->generate_from_linear_ir(reinterpret_cast<const void*>(&jcp)));
}

void Subgraph::SubgraphJitExecutor::init_runtime_params(const std::shared_ptr<CPURuntimeConfig>& cpu_config) {
    // Buffer size should be in CPURtunimeConfig too
    buffer_scratchpad_size = schedule->lowering_result.buffer_scratchpad_size;
    buffer_scratchpad.resize(buffer_scratchpad_size * parallel_get_max_threads(), 0);
    parallel_exec_domain = cpu_config->parallel_domain;
    harness_work_amount = std::accumulate(parallel_exec_domain.cbegin(), parallel_exec_domain.cend(), size_t(1), std::multiplies<size_t>());
}

void Subgraph::SubgraphJitExecutor::exec(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) {
    if (parallel_exec_domain.size() == rank6D) {
        schedule_6d(inMemPtrs, outMemPtrs);
    } else {
        schedule_nt(inMemPtrs, outMemPtrs);
    }
}

#if defined(__linux__) && defined(SNIPPETS_DEBUG_CAPS)
void Subgraph::SubgraphJitExecutor::segfault_detector() {
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

Subgraph::SubgraphJitStaticExecutor::SubgraphJitStaticExecutor(const std::shared_ptr<SubgraphAttrs>& snippet_attrs,
                                                               const std::vector<ptrdiff_t>& start_offset_in,
                                                               const std::vector<ptrdiff_t>& start_offset_out)
    : SubgraphJitExecutor(snippet_attrs), start_offset_in(start_offset_in), start_offset_out(start_offset_out) {
    const auto& cpu_config = ov::as_type_ptr<CPURuntimeConfig>(snippet_attrs->snippet->update_runtime_config());
    generate(snippet_attrs, cpu_config);
    init_runtime_params(cpu_config);
}

inline void Subgraph::SubgraphJitStaticExecutor::update_ptrs(jit_snippets_call_args& call_args,
                                                             const std::vector<MemoryPtr>& srcMemPtrs,
                                                             const std::vector<MemoryPtr>& dstMemPtrs) {
    for (size_t i = 0; i < srcMemPtrs.size(); i++)
        call_args.src_ptrs[i] = srcMemPtrs[i]->getDataAs<const uint8_t>() + start_offset_in[i];

    for (size_t i = 0; i < dstMemPtrs.size(); i++)
        call_args.dst_ptrs[i] = dstMemPtrs[i]->getDataAs<uint8_t>() + start_offset_out[i];

    if (buffer_scratchpad_size > 0) {
        call_args.buffer_scratchpad_ptr =
                reinterpret_cast<uint8_t*>(buffer_scratchpad.data()) + parallel_get_thread_num() * buffer_scratchpad_size;
    }
}

void Subgraph::SubgraphJitStaticExecutor::schedule_6d(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) {
    const auto& callable = schedule->get_callable<kernel>();
    const auto& dom = parallel_exec_domain;

#if defined(__linux__) && defined(SNIPPETS_DEBUG_CAPS)
    segfault_detector();
#endif

    parallel_for5d(dom[0], dom[1], dom[2], dom[3], dom[4],
        [&](size_t d0, size_t d1, size_t d2, size_t d3, size_t d4) {
            jit_snippets_call_args call_args;
            update_ptrs(call_args, inMemPtrs, outMemPtrs);
            size_t indexes[] = {d0, d1, d2, d3, d4};
            callable(&call_args, indexes);
        });
}

void Subgraph::SubgraphJitStaticExecutor::schedule_nt(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) {
    const auto& callable = schedule->get_callable<kernel>();
    const auto& work_size = parallel_exec_domain;

#if defined(__linux__) && defined(SNIPPETS_DEBUG_CAPS)
    segfault_detector();
#endif

    parallel_nt(0, [&](const int ithr, const int nthr) {
        jit_snippets_call_args call_args;
        update_ptrs(call_args, inMemPtrs, outMemPtrs);

        size_t start = 0, end = 0;
        splitter(harness_work_amount, nthr, ithr, start, end);

        std::vector<size_t> indexes(work_size.size() - 1, 0);
        for (size_t iwork = start; iwork < end; ++iwork) {
            size_t tmp = iwork;
            for (ptrdiff_t j = static_cast<ptrdiff_t>(work_size.size()) - 2; j >= 0; j--) {
                indexes[j] = tmp % work_size[j];
                tmp /= work_size[j];
            }

            callable(&call_args, indexes.data());
        }
    });
}

Subgraph::SubgraphJitShapeAgnosticExecutor::SubgraphJitShapeAgnosticExecutor(const std::shared_ptr<SubgraphAttrs>& snippet_attrs)
    : SubgraphJitExecutor(snippet_attrs) {
    const auto& cpu_config = ov::as_type_ptr<CPURuntimeConfig>(snippet_attrs->snippet->update_runtime_config());
    generate(snippet_attrs, cpu_config);
}

void Subgraph::SubgraphJitShapeAgnosticExecutor::schedule_6d(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) {
    OPENVINO_THROW("SubgraphJitShapeAgnosticExecutor doesn't support execution");
}

void Subgraph::SubgraphJitShapeAgnosticExecutor::schedule_nt(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) {
    OPENVINO_THROW("SubgraphJitShapeAgnosticExecutor doesn't support execution");
}

Subgraph::SubgraphJitDynamicSpecializedExecutor::SubgraphJitDynamicSpecializedExecutor(const std::shared_ptr<SubgraphAttrs>& snippet_attrs,
                                                                                       const std::vector<ptrdiff_t>& start_offset_in,
                                                                                       const std::vector<ptrdiff_t>& start_offset_out,
                                                                                       const std::shared_ptr<SubgraphJitShapeAgnosticExecutor>& agnostic)
    : SubgraphJitExecutor(snippet_attrs), start_offset_in(start_offset_in), start_offset_out(start_offset_out) {
    // copy code
    schedule = agnostic->schedule;

    const auto& cpu_config = ov::as_type_ptr<CPURuntimeConfig>(snippet_attrs->snippet->update_runtime_config());
    init_runtime_params(cpu_config);
}

void Subgraph::SubgraphJitDynamicSpecializedExecutor::init_runtime_params(const std::shared_ptr<CPURuntimeConfig>& cpu_config) {
    SubgraphJitExecutor::init_runtime_params(cpu_config);
    data_offsets = cpu_config->io_data_offsets;
    loop_args = cpu_config->loop_args;
}

inline void Subgraph::SubgraphJitDynamicSpecializedExecutor::init_original_ptrs(const std::vector<MemoryPtr>& srcMemPtrs,
                                                                                const std::vector<MemoryPtr>& dstMemPtrs,
                                                                                std::vector<const uint8_t*>& src_ptrs,
                                                                                std::vector<uint8_t*>& dst_ptrs) {
    const auto in_num = srcMemPtrs.size();
    const auto out_num = dstMemPtrs.size();

    src_ptrs.resize(in_num, nullptr);
    dst_ptrs.resize(out_num, nullptr);

    for (size_t i = 0; i < in_num; i++)
        src_ptrs[i] = srcMemPtrs[i]->getDataAs<const uint8_t>() + start_offset_in[i];
    for (size_t i = 0; i < out_num; i++)
        dst_ptrs[i] = dstMemPtrs[i]->getDataAs<uint8_t>() + start_offset_out[i];
}

inline void Subgraph::SubgraphJitDynamicSpecializedExecutor::init_call_args(jit_snippets_call_args& call_args) {
    call_args.register_loops(loop_args);

    if (buffer_scratchpad_size > 0) {
        call_args.buffer_scratchpad_ptr = reinterpret_cast<uint8_t*>(buffer_scratchpad.data()) + parallel_get_thread_num() * buffer_scratchpad_size;
    }
}

inline void Subgraph::SubgraphJitDynamicSpecializedExecutor::update_ptrs(jit_snippets_call_args& call_args,
                                                                        const std::vector<const uint8_t*>& src_ptrs,
                                                                        const std::vector<uint8_t*>& dst_ptrs,
                                                                        const size_t* indexes) const {
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

void Subgraph::SubgraphJitDynamicSpecializedExecutor::schedule_6d(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) {
    const auto& dom = parallel_exec_domain;

    OPENVINO_ASSERT(dom.size() == rank6D, "Incorrect parallel execution domain rank!");
    OPENVINO_ASSERT(data_offsets.size() == inMemPtrs.size() + outMemPtrs.size(), "Incorrect data offset count!");
    OPENVINO_ASSERT(data_offsets.front().size() == rank6D, "Data offsets with invalid ranks detected");

    std::vector<const uint8_t*> src_ptrs;
    std::vector<uint8_t*> dst_ptrs;
    init_original_ptrs(inMemPtrs, outMemPtrs, src_ptrs, dst_ptrs);

#if defined(__linux__) && defined(SNIPPETS_DEBUG_CAPS)
    segfault_detector();
#endif

    // Use parallel_nt instead of parallel_for5d to create thread-local `jit_snippets_call_args` entity
    // with inited buffer scratchpad ptr and loop args only once.
    const auto& callable = schedule->get_callable<dynamic_kernel>();
    parallel_nt(0, [&](const int ithr, const int nthr) {
        jit_snippets_call_args call_args;
        init_call_args(call_args);

        auto func = [&](size_t d0, size_t d1, size_t d2, size_t d3, size_t d4) {
            size_t indexes[] = {d0, d1, d2, d3, d4};
            update_ptrs(call_args, src_ptrs, dst_ptrs, indexes);
            callable(&call_args);
        };

        const size_t work_amount = harness_work_amount;
        if (work_amount == 0)
            return;

        size_t start{0}, end{0};
        splitter(work_amount, nthr, ithr, start, end);

        size_t d0{0}, d1{0}, d2{0}, d3{0}, d4{0};
        parallel_it_init(start, d0, dom[0], d1, dom[1], d2, dom[2], d3, dom[3], d4, dom[4]);
        for (size_t iwork = start; iwork < end; ++iwork) {
            ov::helpers::call_with_args(func, ithr, iwork, d0, d1, d2, d3, d4);
            parallel_it_step(d0, dom[0], d1, dom[1], d2, dom[2], d3, dom[3], d4, dom[4]);
        }
    });
}

void Subgraph::SubgraphJitDynamicSpecializedExecutor::schedule_nt(const std::vector<MemoryPtr>& inMemPtrs, const std::vector<MemoryPtr>& outMemPtrs) {
    const auto& dom = parallel_exec_domain;

    OPENVINO_ASSERT(data_offsets.size() == inMemPtrs.size() + outMemPtrs.size(), "Incorrect data offset count!");
    OPENVINO_ASSERT(data_offsets.front().size() == dom.size(), "Data offsets with invalid ranks detected");

    std::vector<const uint8_t*> src_ptrs;
    std::vector<uint8_t*> dst_ptrs;
    init_original_ptrs(inMemPtrs, outMemPtrs, src_ptrs, dst_ptrs);

#if defined(__linux__) && defined(SNIPPETS_DEBUG_CAPS)
    segfault_detector();
#endif

    const auto& callable = schedule->get_callable<dynamic_kernel>();
    parallel_nt(0, [&](const int ithr, const int nthr) {
        jit_snippets_call_args call_args;
        call_args.register_loops(loop_args);

        size_t start = 0, end = 0;
        splitter(harness_work_amount, nthr, ithr, start, end);

        std::vector<size_t> indexes(dom.size() - 1, 0);
        for (size_t iwork = start; iwork < end; ++iwork) {
            size_t tmp = iwork;
            for (ptrdiff_t j = static_cast<ptrdiff_t>(dom.size()) - 2; j >= 0; j--) {
                indexes[j] = tmp % dom[j];
                tmp /= dom[j];
            }

            update_ptrs(call_args, src_ptrs, dst_ptrs, indexes.data());
            callable(&call_args);
        }
    });
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
