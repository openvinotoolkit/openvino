// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "draft_block_mask.hpp"

#include <cmath>
#include <limits>
#include <memory>

#include "cm_db/experimental_kernel_generator.hpp"
#include "common_utils/jitter.hpp"
#include "impls/cm/primitive_cm_base.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "experimental/primitives/draft_block_mask.hpp"
#include "intel_gpu/runtime/memory.hpp"

namespace ov::intel_gpu::cm {

using namespace cldnn;

namespace {

// Number of work items cooperatively finding the top-k threshold for one (head, Q-temporal-group)
// slice in draft_block_mask_topk_mask.cm; must match the kernel's own TOPK_LOCAL_SIZE default.
constexpr size_t topk_local_size = 32;
constexpr size_t cm_register_file_size = 256;

struct DraftBlockMaskGeometry {
    size_t num_heads = 0;
    size_t seqlen = 0;
    size_t topk = 0;
    size_t batch_size = 1;
    size_t Nq = 0;
    size_t Nk = 0;
    size_t block_s = 0;
    size_t D = 0;
    size_t head_dim = 0;
    // Q-side spatial window count (HWq = Nq / seqlen); also this op's per-temporal-group Q-row
    // count, used by the top-k stage's dispatch/grouping.
    size_t spatial_windows = 0;
    // K-side spatial window count (HWk = Nk / k_groups, k_groups independent of seqlen -- a
    // streaming decoder's accumulated KV cache growing past the current query window). Kept as a
    // separate field from spatial_windows for the two stages' distinct kernel args, but core-level
    // shape_infer()/evaluate() now guarantee local_mask is square, so spatial_windows_k ==
    // spatial_windows whenever both are statically known -- see reference::draft_block_mask.
    size_t spatial_windows_k = 0;
    // True for the 5-input primitive variant (runtime seqlen/topk inputs -- see
    // ov::op::experimental::DraftBlockMask's 5-input constructor).
    bool has_runtime_seqlen_topk = false;
};

DraftBlockMaskGeometry resolve_geometry(const kernel_impl_params& params) {
    const auto desc = params.typed_desc<draft_block_mask>();
    OPENVINO_ASSERT(desc->num_heads > 0, "[GPU][CM] draft_block_mask expects positive num_heads");

    const auto& q_shape = params.input_layouts[DraftBlockMaskInputIdx::Q_WINDOWS].get_partial_shape();
    const auto& k_shape = params.input_layouts[DraftBlockMaskInputIdx::K_WINDOWS].get_partial_shape();
    const auto& mask_shape = params.input_layouts[DraftBlockMaskInputIdx::LOCAL_MASK].get_partial_shape();

    // Only block_s/D (WINDOW_TOKENS/HEAD_DIM) need to be static here: those are the only fields
    // get_jit_constants() reads, and it (like get_arguments_desc()) runs once, at kernel-build time,
    // against the primitive's *declared* layout -- which may still carry dynamic Nq/Nk/HW (FlashVSR's
    // dynamic-H/W dit graph). Nq/Nk/HW are read best-effort (0 if still dynamic) below;
    // get_dispatch_data_func()'s lambdas and get_internal_buffer_descs() are the only callers that
    // need them for real, and those run per actual execution against fully resolved
    // kernel_impl_params -- mirrors BlockSparseAttention's resolve_geometry() (see that file's
    // identical rationale).
    OPENVINO_ASSERT(q_shape.rank().is_static() && q_shape[1].is_static() && q_shape[2].is_static(),
                    "[GPU][CM] draft_block_mask requires a static block_s/D on q_windows");
    OPENVINO_ASSERT(k_shape.rank().is_static() && k_shape[2].is_static(), "[GPU][CM] draft_block_mask requires a static D on k_windows");
    OPENVINO_ASSERT(mask_shape.rank().is_static(), "[GPU][CM] draft_block_mask requires a static-rank local_mask");

    DraftBlockMaskGeometry g;
    g.num_heads = static_cast<size_t>(desc->num_heads);
    g.batch_size = static_cast<size_t>(desc->batch_size);
    g.Nq = q_shape[0].is_static() ? static_cast<size_t>(q_shape[0].get_length()) : 0;
    g.Nk = k_shape[0].is_static() ? static_cast<size_t>(k_shape[0].get_length()) : 0;
    g.block_s = static_cast<size_t>(q_shape[1].get_length());
    g.D = static_cast<size_t>(q_shape[2].get_length());
    OPENVINO_ASSERT(g.D % g.num_heads == 0, "[GPU][CM] draft_block_mask: D must be divisible by num_heads");
    g.head_dim = g.D / g.num_heads;
    g.has_runtime_seqlen_topk = params.input_layouts.size() == 5;

    // spatial_windows (HWq) and seqlen are always derivable from shapes alone (Nq == seqlen*HWq,
    // HWq == local_mask.shape[0]) -- no need to read the `seqlen` input's runtime VALUE even in the
    // 5-input case, since local_mask's shape already carries the same information for free. Falls
    // back to the static `desc->seqlen` attribute only when shapes aren't concrete yet (a
    // build-time-only call in the 3-input case; the 5-input case has no such attribute to fall back
    // to, but neither get_jit_constants() nor get_arguments_desc() actually dereference
    // seqlen/spatial_windows, so an all-zero fallback here is harmless -- only
    // get_dispatch_data_func()'s lambdas need real values, and those always run against concrete,
    // fully-resolved shapes). spatial_windows_k (HWk) is independently derived from local_mask's
    // *second* dim and Nk -- it is NOT assumed equal to spatial_windows (see the field comment).
    const size_t hw_q = mask_shape[0].is_static() ? static_cast<size_t>(mask_shape[0].get_length()) : 0;
    const size_t hw_k = mask_shape[1].is_static() ? static_cast<size_t>(mask_shape[1].get_length()) : 0;
    // Defense-in-depth: core-level shape_infer()/evaluate() already reject a non-square local_mask
    // before this code runs in practice, but assert it locally too in case this kernel_impl_params
    // ever bypasses core validate_and_infer_types().
    OPENVINO_ASSERT(hw_q == 0 || hw_k == 0 || hw_q == hw_k, "[GPU][CM] draft_block_mask requires a square local_mask, got [", hw_q, ", ", hw_k, "]");
    if (hw_q > 0 && g.Nq > 0) {
        g.spatial_windows = hw_q;
        g.seqlen = g.Nq / hw_q;
    } else if (!g.has_runtime_seqlen_topk && desc->seqlen > 0) {
        g.seqlen = static_cast<size_t>(desc->seqlen);
        g.spatial_windows = g.Nq > 0 ? g.Nq / g.seqlen : 0;
    } else {
        g.seqlen = 0;
        g.spatial_windows = 0;
    }
    g.spatial_windows_k = hw_k > 0 ? hw_k : (g.seqlen > 0 && g.Nk > 0 ? g.Nk / g.seqlen : 0);
    // topk: the 3-input (static) primitive carries it as an attribute; the 5-input (runtime) variant
    // has no such attribute -- its real value is read lazily at dispatch time (see
    // read_topk_value(), the sole consumer, called only from DraftBlockMaskTopKGenerator's dispatch
    // func).
    g.topk = g.has_runtime_seqlen_topk ? 0 : static_cast<size_t>(desc->topk);

    return g;
}

// Reads the runtime `topk` input's scalar value off the device. Only called when
// has_runtime_seqlen_topk is true (5-input primitive variant). Mirrors
// BlockSparseAttention::read_scale_value(); unlike that function, topk may be i32 or i64 (both
// accepted by the core op's validation), so the element type is checked at read time.
int32_t read_topk_value(const RuntimeParams& params) {
    const auto topk_mem = params.memory_deps.at(DraftBlockMaskInputIdx::TOPK);
    const auto dtype = topk_mem->get_layout().data_type;
    int64_t value = 0;
    if (dtype == data_types::i32) {
        mem_lock<int32_t, mem_lock_type::read> lock(topk_mem, *params.strm);
        value = lock[0];
    } else {
        mem_lock<int64_t, mem_lock_type::read> lock(topk_mem, *params.strm);
        value = lock[0];
    }
    OPENVINO_ASSERT(value > 0 && value <= std::numeric_limits<int32_t>::max(), "topk out of valid range (1..INT32_MAX)");
    return static_cast<int32_t>(value);
}

enum DraftBlockMaskInternBuffIdx {
    POOLED_Q = 0,    // [Nq, num_heads, head_dim] half -- mean-pooled query blocks
    POOLED_K = 1,    // [Nk, num_heads, head_dim] half -- mean-pooled key blocks
    RAW_SCORES = 2,  // [num_heads, Nq, Nk] float -- score_softmax's own scratch (never read by host/other stages)
    // [num_heads, Nq, Nk] float -- softmax output, consumed by the top-k stage. float (not half):
    // the top-k stage compares raw bit patterns as a monotonic sort key, and normalized softmax
    // probabilities routinely underflow to the SAME half value (most severely 0.0) once Nk is
    // large, clustering ties at the selection threshold and making the number of kept elements
    // unrelated to topk -- see draft_block_mask_topk_mask.cm's file header comment.
    PROBS = 3,
};

void push_int32_scalar(KernelData& kd, int32_t value) {
    scalar_desc desc;
    desc.t = scalar_desc::Types::INT32;
    desc.v.s32 = value;
    kd.params.scalars.push_back(desc);
}

void push_float_scalar(KernelData& kd, float value) {
    scalar_desc desc;
    desc.t = scalar_desc::Types::FLOAT32;
    desc.v.f32 = value;
    kd.params.scalars.push_back(desc);
}

// Mean-pools one of q_windows/k_windows ([N, block_s, D] half) into [N, num_heads, head_dim] half.
// Same kernel entry point is used for both Q and K, bound to different buffers per Stage instance.
class DraftBlockMaskPoolGenerator : public ov::intel_gpu::experimental::cm::ExperimentalKernelGenerator {
public:
    explicit DraftBlockMaskPoolGenerator(bool is_query) : ov::intel_gpu::experimental::cm::ExperimentalKernelGenerator("draft_block_mask_pool", is_query ? "_q" : "_k"), _is_query(is_query) {}

    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + " -Qxcm_register_file_size=" + std::to_string(cm_register_file_size);
    }

    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));

        const auto g = resolve_geometry(params);
        jit.make("NUM_HEADS", g.num_heads);
        jit.make("HEAD_DIM", g.head_dim);
        jit.make("WINDOW_TOKENS", g.block_s);

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& /*params*/) const override {
        Arguments args;
        args.push_back(
            {ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(_is_query ? DraftBlockMaskInputIdx::Q_WINDOWS : DraftBlockMaskInputIdx::K_WINDOWS)});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER,
                        static_cast<uint32_t>(_is_query ? DraftBlockMaskInternBuffIdx::POOLED_Q : DraftBlockMaskInternBuffIdx::POOLED_K)});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        const bool is_query = _is_query;
        return DispatchDataFunc{[is_query](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* /*rt_params*/) {
            OPENVINO_ASSERT(!params.is_dynamic());
            const auto g = resolve_geometry(params);
            auto& wgs = kd.params.workGroups;
            wgs.global = {is_query ? g.Nq : g.Nk, g.num_heads, 1};
            wgs.local = {1, 1, 1};
        }};
    }

private:
    bool _is_query;
};

// Block-level QK^T on pooled Q/K, plus the tiled static local-spatial-window additive mask and
// row-wise softmax (fused, per draft_block_mask_score_softmax.cm's design notes).
class DraftBlockMaskScoreSoftmaxGenerator : public ov::intel_gpu::experimental::cm::ExperimentalKernelGenerator {
public:
    DraftBlockMaskScoreSoftmaxGenerator() : ov::intel_gpu::experimental::cm::ExperimentalKernelGenerator("draft_block_mask_score_softmax") {}

    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + " -Qxcm_register_file_size=" + std::to_string(cm_register_file_size);
    }

    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));

        const auto g = resolve_geometry(params);
        jit.make("NUM_HEADS", g.num_heads);
        jit.make("HEAD_DIM", g.head_dim);

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, DraftBlockMaskInternBuffIdx::POOLED_Q});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, DraftBlockMaskInternBuffIdx::POOLED_K});
        args.push_back({ArgumentDescriptor::Types::INPUT, DraftBlockMaskInputIdx::LOCAL_MASK});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, DraftBlockMaskInternBuffIdx::RAW_SCORES});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, DraftBlockMaskInternBuffIdx::PROBS});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // num_q_windows
        args.push_back({ArgumentDescriptor::Types::SCALAR, 1});  // num_k_windows
        args.push_back({ArgumentDescriptor::Types::SCALAR, 2});  // spatial_windows (HWq)
        args.push_back({ArgumentDescriptor::Types::SCALAR, 3});  // spatial_windows_k (HWk)
        args.push_back({ArgumentDescriptor::Types::SCALAR, 4});  // scale
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* /*rt_params*/) {
            OPENVINO_ASSERT(!params.is_dynamic());
            const auto g = resolve_geometry(params);
            auto& wgs = kd.params.workGroups;
            wgs.global = {g.Nq, g.num_heads, 1};
            wgs.local = {1, 1, 1};

            kd.params.scalars.clear();
            push_int32_scalar(kd, static_cast<int32_t>(g.Nq));
            push_int32_scalar(kd, static_cast<int32_t>(g.Nk));
            push_int32_scalar(kd, static_cast<int32_t>(g.spatial_windows));
            push_int32_scalar(kd, static_cast<int32_t>(g.spatial_windows_k));
            push_float_scalar(kd, 1.0f / std::sqrt(static_cast<float>(g.head_dim)));
        }};
    }
};

// Global top-k over the flattened (spatial_windows * num_k_windows) elements of one
// (head, Q-temporal-group), per draft_block_mask_topk_mask.cm's 4-pass MSD radix-select.
class DraftBlockMaskTopKGenerator : public ov::intel_gpu::experimental::cm::ExperimentalKernelGenerator {
public:
    DraftBlockMaskTopKGenerator() : ov::intel_gpu::experimental::cm::ExperimentalKernelGenerator("draft_block_mask_topk_mask") {}

    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + " -Qxcm_register_file_size=" + std::to_string(cm_register_file_size);
    }

    [[nodiscard]] JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));

        const auto g = resolve_geometry(params);
        jit.make("NUM_HEADS", g.num_heads);
        jit.make("TOPK_LOCAL_SIZE", topk_local_size);

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const kernel_impl_params& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, DraftBlockMaskInternBuffIdx::PROBS});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // num_q_windows
        args.push_back({ArgumentDescriptor::Types::SCALAR, 1});  // num_k_windows
        args.push_back({ArgumentDescriptor::Types::SCALAR, 2});  // spatial_windows
        args.push_back({ArgumentDescriptor::Types::SCALAR, 3});  // topk
        args.push_back({ArgumentDescriptor::Types::SCALAR, 4});  // batch_size
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* /*rt_params*/) {
            OPENVINO_ASSERT(!params.is_dynamic());
            const auto g = resolve_geometry(params);
            auto& wgs = kd.params.workGroups;
            wgs.global = {g.num_heads, g.seqlen, topk_local_size};
            wgs.local = {1, 1, topk_local_size};

            kd.params.scalars.clear();
            push_int32_scalar(kd, static_cast<int32_t>(g.Nq));
            push_int32_scalar(kd, static_cast<int32_t>(g.Nk));
            push_int32_scalar(kd, static_cast<int32_t>(g.spatial_windows));
            push_int32_scalar(kd, g.has_runtime_seqlen_topk ? read_topk_value(params) : static_cast<int32_t>(g.topk));
            // mask_out is [batch_size, num_heads, Nq, Nk]; the mask itself is batch-independent
            // (computed once from spatially-static Q/K windows), so it's tiled across batch here
            // rather than recomputed -- see reference::draft_block_mask's identical step 4.
            push_int32_scalar(kd, static_cast<int32_t>(g.batch_size));
        }};
    }
};

class DraftBlockMaskImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::DraftBlockMaskImpl)

    Stage::Ptr pool_q = make_stage<DraftBlockMaskPoolGenerator>(true);
    Stage::Ptr pool_k = make_stage<DraftBlockMaskPoolGenerator>(false);
    Stage::Ptr score_softmax = make_stage<DraftBlockMaskScoreSoftmaxGenerator>();
    Stage::Ptr topk_mask = make_stage<DraftBlockMaskTopKGenerator>();

    DraftBlockMaskImpl() : PrimitiveImplCM(DraftBlockMaskImplementationManager::get_type_info_static()) {}
    DraftBlockMaskImpl(const program_node& /*node*/, const RuntimeParams& params) : DraftBlockMaskImpl() {
        add_stage(pool_q, params);
        add_stage(pool_k, params);
        add_stage(score_softmax, params);
        add_stage(topk_mask, params);
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        const auto g = resolve_geometry(params);
        std::vector<BufferDescriptor> buffers;
        buffers.emplace_back(g.Nq * g.num_heads * g.head_dim, ov::element::f16);  // POOLED_Q
        buffers.emplace_back(g.Nk * g.num_heads * g.head_dim, ov::element::f16);  // POOLED_K
        buffers.emplace_back(g.num_heads * g.Nq * g.Nk, ov::element::f32);        // RAW_SCORES
        buffers.emplace_back(g.num_heads * g.Nq * g.Nk, ov::element::f32);        // PROBS
        return buffers;
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<DraftBlockMaskImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> DraftBlockMaskImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<draft_block_mask>());
    return std::make_unique<DraftBlockMaskImpl>(node, params);
}

}  // namespace ov::intel_gpu::cm

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::DraftBlockMaskImpl)
