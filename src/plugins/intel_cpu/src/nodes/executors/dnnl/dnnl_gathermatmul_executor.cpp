// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_gathermatmul_executor.hpp"

#include <oneapi/dnnl/dnnl_common_types.h>
#include <oneapi/dnnl/dnnl_threadpool.h>
#include <oneapi/dnnl/dnnl_types.h>

#include <algorithm>
#include <atomic>
#include <common/primitive_hashing.hpp>
#include <common/primitive_hashing_utils.hpp>
#include <common/utils.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <oneapi/dnnl/dnnl_threadpool.hpp>
#include <oneapi/dnnl/dnnl_threadpool_iface.hpp>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "nodes/common/blocked_desc_creator.h"
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include <cpu/x64/cpu_isa_traits.hpp>
#endif
#include "cpu_memory.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/common/offset_helper.hpp"
#include "nodes/executors/dnnl/dnnl_utils.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/gathermatmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "thread_pool_imp.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

// ---- InnerProductKey ---------------------------------------------------------

struct InnerProductKey {
    dnnl::memory::desc src_md;
    dnnl::memory::desc weights_md;
    dnnl::memory::desc bias_md;
    VectorDims scale_shape;
    VectorDims zp_shape;
    dnnl::memory::data_type zp_dt = dnnl::memory::data_type::f32;
    size_t src_dyn_quant_group_size = 0;
    // One-core mode: the primitive is built for a single thread and takes its scratchpad from the caller, so
    // several copies can run at the same time on different threads.
    bool single_thread = false;

    [[nodiscard]] size_t hash() const {
        using namespace dnnl::impl;
        using namespace dnnl::impl::primitive_hashing;

        size_t seed = 0;
        seed = hash_combine(seed, get_md_hash(*src_md.get()));
        seed = hash_combine(seed, get_md_hash(*weights_md.get()));
        seed = hash_combine(seed, get_md_hash(*bias_md.get()));
        seed = get_vector_hash(seed, scale_shape);
        seed = get_vector_hash(seed, zp_shape);
        seed = hash_combine(seed, static_cast<size_t>(zp_dt));
        seed = hash_combine(seed, src_dyn_quant_group_size);
        seed = hash_combine(seed, single_thread);
        return seed;
    }

    bool operator==(const InnerProductKey& rhs) const {
        return src_md == rhs.src_md && weights_md == rhs.weights_md && bias_md == rhs.bias_md &&
               scale_shape == rhs.scale_shape && zp_shape == rhs.zp_shape && zp_dt == rhs.zp_dt &&
               src_dyn_quant_group_size == rhs.src_dyn_quant_group_size && single_thread == rhs.single_thread;
    }
};

static std::pair<size_t, dnnl::memory::data_type> dynQuantParams(uint64_t dq_group_size,
                                                                 ov::element::Type wei_precision,
                                                                 ov::element::Type src_precision,
                                                                 ov::element::Type zp_precision,
                                                                 bool zp_non_negative_i8,
                                                                 size_t K,
                                                                 const VectorDims& scale_shape,
                                                                 const VectorDims& zp_shape) {
    const auto disabled = std::make_pair(size_t{0}, dnnl::memory::data_type::f32);
    if (dq_group_size == 0 || dq_group_size == std::numeric_limits<uint64_t>::max() || dq_group_size % 32 != 0) {
        return disabled;
    }
    // u3 weights come with i8 zero points. Dynamic quantization converts the zero points to u8, which is
    // lossless only when none of them is negative.
    const bool zp_ok = any_of(zp_precision,
                              ov::element::u8,
                              ov::element::u2,
                              ov::element::u3,
                              ov::element::u4,
                              ov::element::dynamic) ||
                       (zp_precision == ov::element::i8 && zp_non_negative_i8);
    const bool is_u3_weights = wei_precision == ov::element::u3;
    if (!is_u3_weights || !zp_ok) {
        return disabled;
    }
    if (K < dq_group_size) {
        return disabled;
    }
    for (const auto* shape : {&scale_shape, &zp_shape}) {
        if (shape->size() >= 2 && (*shape)[1] > 1 && (K / (*shape)[1]) % dq_group_size) {
            return disabled;
        }
    }
    if (src_precision == ov::element::bf16) {
        if (!ov::with_cpu_x86_bfloat16() || ov::with_cpu_x86_avx512_core_amx()) {
            return disabled;
        }
    } else if (!ov::with_cpu_x86_avx2_vnni() && !ov::with_cpu_x86_avx512_core_vnni()) {
        return disabled;
    }
    const auto zp_dt =
        zp_precision == ov::element::dynamic ? dnnl::memory::data_type::f32 : dnnl::memory::data_type::u8;
    return {static_cast<size_t>(dq_group_size), zp_dt};
}

#if OV_THREAD == OV_THREAD_TBB_ADAPTIVE
namespace {
// oneDNN thread pool that runs all work on the calling thread. A one-core inner product executes on a stream
// over this pool, so it never starts a parallel region of its own.
class OneThreadPool : public dnnl::threadpool_interop::threadpool_iface {
public:
    [[nodiscard]] int get_num_threads() const override {
        return 1;
    }
    [[nodiscard]] bool get_in_parallel() const override {
        return true;
    }
    [[nodiscard]] uint64_t get_flags() const override {
        return 0;
    }
    void parallel_for(int n, const std::function<void(int, int)>& fn) override {
        for (int i = 0; i < n; i++) {
            fn(i, n);
        }
    }
    void wait() override {}
};
}  // namespace
#endif

// ---- InnerProduct (oneDNN inner_product wrapper) ----------------------------

class GatherMatmulDnnlExecutor::InnerProduct {
public:
    InnerProduct() = delete;
    InnerProduct(const InnerProduct&) = delete;
    InnerProduct(InnerProduct&&) = delete;
    InnerProduct& operator=(const InnerProduct&) = delete;
    InnerProduct& operator=(InnerProduct&&) = delete;

    InnerProduct(const dnnl::engine& eng, const std::shared_ptr<ThreadPool>& threadPool, const InnerProductKey& key)
        : m_eng(eng),
          m_stream(make_stream(eng, threadPool)) {
        const auto& src_md = key.src_md;
        const auto& weights_md = key.weights_md;
        auto scale_shape = key.scale_shape;
        auto zp_shape = key.zp_shape;

        const auto weiDims = weights_md.get_dims();
        const auto K = weiDims[1];
        const auto N = weiDims[0];
        const auto M = src_md.get_dims()[0];

        if (!scale_shape.empty()) {
            if (all_of(1U, scale_shape.size(), scale_shape[0])) {
                scale_shape.push_back(1);
            }
            OPENVINO_ASSERT(scale_shape.size() == 2, "Unsupported scale shape ", vec2str(scale_shape));
            const auto K_groups = scale_shape.back();
            OPENVINO_ASSERT((K % K_groups) == 0, "Incompatible number of groups ", K_groups, " for K ", K);
            init_w_scales(scale_shape);
            if (!zp_shape.empty()) {
                if (all_of(1U, zp_shape.size(), zp_shape[0])) {
                    zp_shape.push_back(1);
                }
                OPENVINO_ASSERT(zp_shape.size() == 2, "Unsupported zero points shape ", vec2str(zp_shape));
                init_w_zp(zp_shape, key.src_dyn_quant_group_size > 0 ? key.zp_dt : dnnl::memory::data_type::f32);
            }
        }

        if (key.src_dyn_quant_group_size > 0) {
            m_attr.set_src_dyn_quant_params(key.src_dyn_quant_group_size);
        }

        m_input_md = src_md;
        m_output_md =
            dnnl::memory::desc(dnnl::memory::dims({M, N}), src_md.get_data_type(), dnnl::memory::format_tag::ab);

        const auto& bias_md = key.bias_md;
        m_bias_md = bias_md;

        if (key.single_thread) {
            m_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        }
        auto create_prim_desc = [&]() {
            return dnnl::inner_product_forward::primitive_desc(eng,
                                                               dnnl::prop_kind::forward_inference,
                                                               m_input_md,
                                                               weights_md,
                                                               bias_md,
                                                               m_output_md,
                                                               m_attr);
        };
        dnnl::inner_product_forward::primitive_desc ip_prim_desc;
        if (key.single_thread) {
#if OV_THREAD == OV_THREAD_TBB_ADAPTIVE
            // oneDNN picks the kernel and its work split for the concurrency of the creating thread (a thread-local
            // value), so the primitive is created with a concurrency of one and the previous value is restored.
            int prev_concurrency = 0;
            dnnl_threadpool_interop_get_max_concurrency(&prev_concurrency);
            dnnl_threadpool_interop_set_max_concurrency(1);
            try {
                ip_prim_desc = create_prim_desc();
            } catch (...) {
                dnnl_threadpool_interop_set_max_concurrency(prev_concurrency);
                throw;
            }
            dnnl_threadpool_interop_set_max_concurrency(prev_concurrency);
            m_scratch_md = ip_prim_desc.scratchpad_desc();
#else
            OPENVINO_THROW("GatherMatmul: the one-core inner product needs the oneDNN threadpool runtime");
#endif
        } else {
            ip_prim_desc = create_prim_desc();
        }

        m_impl_type = parse_impl_name(ip_prim_desc.impl_info_str());
        m_wei_md = ip_prim_desc.weights_desc();
        m_prim = dnnl::inner_product_forward(ip_prim_desc);

        dnnl::memory inp_memory(m_input_md, eng, DNNL_MEMORY_NONE);
        dnnl::memory out_memory(m_output_md, eng, DNNL_MEMORY_NONE);
        dnnl::memory wei_memory(m_wei_md, eng, DNNL_MEMORY_NONE);
        dnnl::memory bias_memory;
        if (!bias_md.is_zero()) {
            bias_memory = dnnl::memory(bias_md, eng, DNNL_MEMORY_NONE);
        }
        dnnl::memory scale_memory;
        if (!scale_shape.empty()) {
            scale_memory = dnnl::memory(m_scale_md, eng, DNNL_MEMORY_NONE);
        }
        dnnl::memory zp_memory;
        if (!zp_shape.empty()) {
            zp_memory = dnnl::memory(m_zp_md, eng, DNNL_MEMORY_NONE);
        }
        m_args = make_args(inp_memory, out_memory, wei_memory, bias_memory, scale_memory, zp_memory);
    }

    void exec(void* src, void* dst, void* weight, void* bias = nullptr, void* scale = nullptr, void* zp = nullptr) {
        m_args[DNNL_ARG_SRC].set_data_handle(src);
        m_args[DNNL_ARG_DST].set_data_handle(dst);
        m_args[DNNL_ARG_WEIGHTS].set_data_handle(weight);
        if (bias) {
            m_args[DNNL_ARG_BIAS].set_data_handle(bias);
        }
        if (scale) {
            m_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS].set_data_handle(scale);
        }
        if (zp) {
            m_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS].set_data_handle(zp);
        }
        m_prim.execute(m_stream, m_args);
    }

    // Size of the scratchpad that exec_single_thread() needs. Zero for a full-machine inner product.
    [[nodiscard]] size_t scratchpad_size() const {
        return m_scratch_md.get_size();
    }

    // Runs a one-core inner product (key.single_thread) on the calling thread. Several threads may call it on the
    // same object at the same time: every call builds its own arguments, and each thread uses its own stream and
    // its own scratchpad of scratchpad_size() bytes.
    void exec_single_thread(void* src, void* dst, void* weight, void* bias, void* scale, void* zp, void* scratch)
        const {
#if OV_THREAD == OV_THREAD_TBB_ADAPTIVE
        thread_local OneThreadPool one_thread_pool;
        thread_local dnnl::stream one_thread_stream;
        if (!one_thread_stream || one_thread_stream.get_engine() != m_eng) {
            one_thread_stream = dnnl::threadpool_interop::make_stream(m_eng, &one_thread_pool);
        }
        std::unordered_map<int, dnnl::memory> args;
        args.emplace(DNNL_ARG_SRC, dnnl::memory(m_input_md, m_eng, src));
        args.emplace(DNNL_ARG_DST, dnnl::memory(m_output_md, m_eng, dst));
        args.emplace(DNNL_ARG_WEIGHTS, dnnl::memory(m_wei_md, m_eng, weight));
        if (bias && !m_bias_md.is_zero()) {
            args.emplace(DNNL_ARG_BIAS, dnnl::memory(m_bias_md, m_eng, bias));
        }
        if (scale && !m_scale_md.is_zero()) {
            args.emplace(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(m_scale_md, m_eng, scale));
        }
        if (zp && !m_zp_md.is_zero()) {
            args.emplace(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, dnnl::memory(m_zp_md, m_eng, zp));
        }
        if (!m_scratch_md.is_zero()) {
            args.emplace(DNNL_ARG_SCRATCHPAD, dnnl::memory(m_scratch_md, m_eng, scratch));
        }
        m_prim.execute(one_thread_stream, args);
#else
        OPENVINO_THROW("GatherMatmul: the one-core inner product needs the oneDNN threadpool runtime");
#endif
    }

    [[nodiscard]] dnnl::memory::desc get_weights_md() const {
        return m_wei_md;
    }
    [[nodiscard]] dnnl::memory::desc get_scale_md() const {
        return m_scale_md;
    }
    [[nodiscard]] dnnl::memory::desc get_zp_md() const {
        return m_zp_md;
    }
    [[nodiscard]] impl_desc_type get_impl_type() const {
        return m_impl_type;
    }

private:
    void init_w_scales(const VectorDims& scale_shape) {
        constexpr auto data_type = dnnl::memory::data_type::f32;
        const auto scale_dims = DnnlExtensionUtils::convertToDnnlDims(scale_shape);
        m_attr.set_scales_dims(DNNL_ARG_WEIGHTS, scale_dims, data_type);
        m_scale_md = dnnl::memory::desc(scale_dims, data_type, dnnl::memory::format_tag::ba);
    }

    void init_w_zp(const VectorDims& zp_shape, dnnl::memory::data_type data_type) {
        const auto zp_dims = DnnlExtensionUtils::convertToDnnlDims(zp_shape);
        m_attr.set_zero_points_dims(DNNL_ARG_WEIGHTS, zp_dims, data_type);
        m_zp_md = dnnl::memory::desc(zp_dims, data_type, dnnl::memory::format_tag::ba);
    }

    static std::unordered_map<int, dnnl::memory> make_args(dnnl::memory& src,
                                                           dnnl::memory& dst,
                                                           dnnl::memory& weight,
                                                           dnnl::memory& bias,
                                                           dnnl::memory& scale,
                                                           dnnl::memory& zp) {
        std::unordered_map<int, dnnl::memory> args;
        args.insert({DNNL_ARG_SRC, src});
        args.insert({DNNL_ARG_WEIGHTS, weight});
        args.insert({DNNL_ARG_DST, dst});
        if (bias) {
            args.insert({DNNL_ARG_BIAS, bias});
        }
        if (scale) {
            args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale});
        }
        if (zp) {
            args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp});
        }
        return args;
    }

    dnnl::engine m_eng;
    dnnl::stream m_stream;
    dnnl::primitive m_prim;
    dnnl::memory::desc m_input_md;
    dnnl::memory::desc m_output_md;
    dnnl::memory::desc m_wei_md;
    dnnl::memory::desc m_bias_md;
    dnnl::memory::desc m_scale_md;
    dnnl::memory::desc m_zp_md;
    dnnl::memory::desc m_scratch_md;
    dnnl::primitive_attr m_attr;
    std::unordered_map<int, dnnl::memory> m_args;
    impl_desc_type m_impl_type = impl_desc_type::unknown;
};

namespace {

// ---- normalizeM helper -------------------------------------------------------

Dim normalizeM(Dim M) {
    if (M < 512) {
        M = rnd_up(M, 16);
    } else if (M < 1024) {
        M = rnd_up(M, 32);
    } else {
        M = rnd_up(M, 256);
    }
    return M;
}

// ---- Expert-parallel path (bf16 AMX) ----------------------------------------

// The expert-parallel path cuts the rows routed to each gather index into tasks of at most this many rows, and
// every thread runs whole tasks with a one-core GEMM. Small tasks keep the threads busy when the row counts per
// gather index differ a lot.
constexpr size_t kExpertRowsPerTask = 64;
// normalizeM() rounds a task of at most kExpertRowsPerTask rows up to a multiple of 16, so the tasks need one
// one-core GEMM per height 16, 32, 48 and 64.
constexpr size_t kTaskHeightStep = 16;
constexpr size_t kTaskHeights = kExpertRowsPerTask / kTaskHeightStep;

// ---- GatherMatmulDnnlExecutor -----------------------------------------------

dnnl::memory::desc makeBiasMd(dnnl::memory::dim N, const MemoryPtr& biasMem) {
    if (!biasMem || biasMem->getDesc().empty()) {
        return {};
    }
    const auto bias_precision = biasMem->getDesc().getPrecision();
    return dnnl::memory::desc(dnnl::memory::dims({N}),
                              DnnlExtensionUtils::ElementTypeToDataType(bias_precision),
                              dnnl::memory::format_tag::a);
}
}  // namespace

bool GatherMatmulDnnlExecutor::supports([[maybe_unused]] const GatherMatmulConfig& config) {
#ifdef OPENVINO_ARCH_X86_64
    // Allow empty (dynamic) src descriptor — actual type is resolved at createPrimitive time
    if ((config.descs.count(ARG_SRC) != 0U) && !config.descs.at(ARG_SRC)->empty()) {
        const auto src_prc = config.descs.at(ARG_SRC)->getPrecision();
        if (!any_of(src_prc, ov::element::f32, ov::element::bf16)) {
            return false;
        }
    }
    // For compressed (int) weights, require AVX2
    if ((config.descs.count(ARG_WEI) != 0U) && !config.descs.at(ARG_WEI)->empty()) {
        const auto wei_prc = config.descs.at(ARG_WEI)->getPrecision();
        if (any_of(wei_prc, ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4, ov::element::u3)) {
            if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
                return false;
            }
        }
    }
    return true;
#else
    return false;
#endif
}

GatherMatmulDnnlExecutor::GatherMatmulDnnlExecutor(const GatherMatmulAttrs& attrs,
                                                   const MemoryArgs& memory,
                                                   const ExecutorContext::CPtr& context)
    : m_context(context) {
    const auto& weightsMemory = memory.at(ARG_WEI);
    const auto& srcMemory = memory.at(ARG_SRC);

    auto src_precision = srcMemory->getDesc().getPrecision();
    auto weights_precision = weightsMemory->getDesc().getPrecision();
#ifdef OPENVINO_ARCH_X86_64
    m_bf16AmxMode =
        (src_precision == ov::element::bf16 && dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_amx));
#endif

    const auto& weiDims = weightsMemory->getShape().getStaticDims();
    const dnnl::memory::dim N = weiDims[weiDims.size() - 2];
    const dnnl::memory::dim K = weiDims[weiDims.size() - 1];

    // Determine scale/zp shapes (per-group, removing the leading batch/gather dim)
    VectorDims scale_shape{};
    VectorDims zp_shape{};

    const auto& scalesMem = memory.at(ARG_SRC_3);
    if (scalesMem && !scalesMem->getDesc().empty()) {
        const auto& fullScalesShape = scalesMem->getShape().getStaticDims();
        if (1 == fullScalesShape.size()) {
            OPENVINO_ASSERT(fullScalesShape[0] == 1, "Expect broadcastable scales shape.");
            scale_shape.push_back(fullScalesShape[0]);
        } else {
            scale_shape.assign(fullScalesShape.begin() + 1, fullScalesShape.end());
        }
    }

    const auto& zpMem = memory.at(ARG_SRC_4);
    ov::element::Type zp_precision = ov::element::dynamic;
    bool zp_non_negative_i8 = false;
    if (zpMem && !zpMem->getDesc().empty()) {
        zp_precision = zpMem->getDesc().getPrecision();
        zp_non_negative_i8 = zp_precision == ov::element::i8 && utils::isNonNegativeI8(*zpMem);
        const auto& fullZpShape = zpMem->getShape().getStaticDims();
        if (1 == fullZpShape.size()) {
            OPENVINO_ASSERT(fullZpShape[0] == 1, "Expect broadcastable zero points shape.");
            zp_shape.push_back(fullZpShape[0]);
        } else {
            zp_shape.assign(fullZpShape.begin() + 1, fullZpShape.end());
        }
    }

    dnnl::memory::desc src_md({1, K},
                              DnnlExtensionUtils::ElementTypeToDataType(src_precision),
                              dnnl::memory::format_tag::ab);
    dnnl::memory::desc weights_md({N, K},
                                  DnnlExtensionUtils::ElementTypeToDataType(weights_precision),
                                  dnnl::memory::format_tag::any);

    const auto [src_dyn_quant_group_size, zp_dt] = dynQuantParams(attrs.dynamicQuantizationGroupSize,
                                                                  weights_precision,
                                                                  src_precision,
                                                                  zp_precision,
                                                                  zp_non_negative_i8,
                                                                  static_cast<size_t>(K),
                                                                  scale_shape,
                                                                  zp_shape);

    InnerProductKey key{src_md,
                        weights_md,
                        makeBiasMd(N, memory.at(ARG_BIAS)),
                        scale_shape,
                        zp_shape,
                        zp_dt,
                        src_dyn_quant_group_size};

    const auto& eng = context->getEngine();
    const auto threadPool = context->getThreadPool();
    auto cache = context->getRuntimeCache();
    std::tie(m_gemvImpl, std::ignore) = cache->getOrCreate(key, [&eng, &threadPool](const InnerProductKey& k) {
        return std::make_shared<InnerProduct>(eng, threadPool, k);
    });

    // The M > 1 GEMMs solve the GEMV problem on the weights layout the GEMV selected (the weights are
    // repacked to it below); createGemm() sets the row count.
    key.weights_md = m_gemvImpl->get_weights_md();
    m_gemmKey = std::make_shared<const InnerProductKey>(std::move(key));

    // Repack weights: convert from [G, K, N] to [G, (packed_N, K)] format expected by oneDNN
    auto gemvWeightsDesc =
        MemoryDescUtils::convertToBlockedMemoryDesc(DnnlExtensionUtils::makeDescriptor(m_gemvImpl->get_weights_md()));

    auto addBatchDim = [](const BlockedMemoryDescPtr& desc, size_t batchDim) -> DnnlMemoryDescPtr {
        const auto& weightsDims = desc->getShape().getStaticDims();
        const auto& weightsBlockDims = desc->getBlockDims();
        const auto& weightsOrder = desc->getOrder();
        VectorDims newDims = {batchDim};
        newDims.insert(newDims.end(), weightsDims.begin(), weightsDims.end());
        VectorDims newBlockDims = {batchDim};
        newBlockDims.insert(newBlockDims.end(), weightsBlockDims.begin(), weightsBlockDims.end());
        VectorDims newOrder(weightsOrder.size() + 1);
        newOrder[0] = 0;
        for (size_t i = 0; i < weightsOrder.size(); i++) {
            newOrder[i + 1] = weightsOrder[i] + 1;
        }
        auto targetDesc =
            std::make_shared<CpuBlockedMemoryDesc>(desc->getPrecision(), Shape(newDims), newBlockDims, newOrder);
        return MemoryDescUtils::convertToDnnlMemoryDesc(targetDesc);
    };

    auto targetWeightsDesc = addBatchDim(gemvWeightsDesc, weiDims[0]);
    auto srcWeightsDesc = MemoryDescUtils::convertToDnnlMemoryDesc(weightsMemory->getDescPtr());

    m_weightsMemory = utils::prepareWeightsMemory(srcWeightsDesc,
                                                  targetWeightsDesc,
                                                  weightsMemory,
                                                  eng,
                                                  cache,
                                                  context->getWeightsCache(),
                                                  context->getPrivateWeightCache(),
                                                  threadPool);

    if (scalesMem && !scale_shape.empty()) {
        auto expectedScaleMemDesc =
            MemoryDescUtils::convertToDnnlMemoryDesc(DnnlExtensionUtils::makeDescriptor(m_gemvImpl->get_scale_md()));
        const auto& scDims = scalesMem->getShape().getStaticDims();
        expectedScaleMemDesc =
            addBatchDim(MemoryDescUtils::convertToBlockedMemoryDesc(expectedScaleMemDesc), scDims[0]);
        if (expectedScaleMemDesc->isCompatible(scalesMem->getDesc())) {
            m_scalesMemory = std::const_pointer_cast<IMemory>(scalesMem);
        } else {
            m_scalesMemory = std::make_shared<Memory>(eng, expectedScaleMemDesc);
            m_scalesMemory->load(*scalesMem, false, false);
        }
    }

    if (zpMem && !zp_shape.empty()) {
        auto expectedZpMemDesc =
            MemoryDescUtils::convertToDnnlMemoryDesc(DnnlExtensionUtils::makeDescriptor(m_gemvImpl->get_zp_md()));
        const auto& zpDims = zpMem->getShape().getStaticDims();
        expectedZpMemDesc = addBatchDim(MemoryDescUtils::convertToBlockedMemoryDesc(expectedZpMemDesc), zpDims[0]);
        if (expectedZpMemDesc->isCompatible(zpMem->getDesc())) {
            m_zpMemory = std::const_pointer_cast<IMemory>(zpMem);
        } else {
            m_zpMemory = std::make_shared<Memory>(eng, expectedZpMemDesc);
            m_zpMemory->load(*zpMem, false, false);
        }
    }

    m_implType = m_gemvImpl->get_impl_type();
}

bool GatherMatmulDnnlExecutor::update(const MemoryArgs& memory) {
    const auto& srcMem = memory.at(ARG_SRC);
    const auto& srcShape = srcMem->getStaticDims();
    // srcShape is [B, M, K]
    if (Dim{1} == srcShape[1]) {
        // If M is 1, we can skip the temporary buffer and execute GEMV in-place on the src buffer
        m_implType = m_gemvImpl->get_impl_type();
        return true;
    }
    const Dim M = normalizeM(srcShape[1]);
    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    const auto srcPrc = srcMem->getDesc().getPrecision();

    const auto& dstMem = memory.at(ARG_DST);
    const auto& dstShape = dstMem->getStaticDims();

    m_tmpInputDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(srcPrc, Shape({M, srcShape[2]}));
    m_tmpOutputDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(srcPrc, Shape({M, dstShape[2]}));

    const size_t srcSize = rnd_up(m_tmpInputDesc->getCurrentMemSize(), 64);
    size_t totalSize = srcSize + m_tmpOutputDesc->getCurrentMemSize();
    m_expertParallelOk = m_bf16AmxMode && initExpertParallel(srcShape[2], dstShape[2], srcPrc);
    if (m_expertParallelOk) {
        // Only one of the two paths runs in an execute() call, so they share one scratchpad region.
        totalSize = std::max(totalSize, m_epSlots * m_epSlotBytes);
    }
    auto scratchPadDesc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(ov::element::u8, Shape({totalSize}));
    m_tmpInpBuffer = m_context->getScratchPad()->createScratchPadMem(scratchPadDesc);

    // The GEMMs are created in execute(), each sized to the rows its gather index receives, so the oneDNN
    // primitive that will run is not known here. DNNL_VERBOSE reports the implementation.
    m_implType = impl_desc_type::unknown;
    return true;
}

bool GatherMatmulDnnlExecutor::initExpertParallel(size_t K, size_t N, const ov::element::Type& precision) {
#if OV_THREAD == OV_THREAD_TBB_ADAPTIVE
    if (!m_epGemmsTried) {
        m_epGemmsTried = true;
        std::vector<InnerProductPtr> gemms;
        try {
            for (size_t i = 1; i <= kTaskHeights; i++) {
                auto gemm = createGemm(i * kTaskHeightStep, true);
                // The weights are packed once, for the layout the GEMV selected. A one-core GEMM that wants
                // another layout cannot use them, so the node keeps the per-gather-index loop.
                if (gemm->get_weights_md() != m_gemmKey->weights_md) {
                    return false;
                }
                gemms.push_back(std::move(gemm));
            }
        } catch (const dnnl::error&) {
            // No one-core implementation for this problem: keep the per-gather-index loop.
            return false;
        }
        m_epGemms = std::move(gemms);
    }
    if (m_epGemms.empty()) {
        return false;
    }

    size_t max_scratchpad = 0;
    for (const auto& gemm : m_epGemms) {
        max_scratchpad = std::max(max_scratchpad, gemm->scratchpad_size());
    }
    // One region per slot: the gathered input rows, the output rows, and the GEMM's scratchpad. The size does not
    // depend on the input length, because a task never has more than kExpertRowsPerTask rows.
    m_epInBytes = rnd_up(kExpertRowsPerTask * K * precision.size(), 64);
    m_epOutBytes = rnd_up(kExpertRowsPerTask * N * precision.size(), 64);
    m_epSlotBytes = m_epInBytes + m_epOutBytes + rnd_up(max_scratchpad, 64);
    // execute() runs exactly this many slots, so a slot index always names a valid region, whatever number of
    // threads the parallel region gets.
    m_epSlots = static_cast<size_t>(parallel_get_max_threads());
    return m_epSlots > 0;
#else
    (void)K;
    (void)N;
    (void)precision;
    return false;
#endif
}

GatherMatmulDnnlExecutor::InnerProductPtr GatherMatmulDnnlExecutor::createGemm(size_t rows, bool single_thread) const {
    OPENVINO_ASSERT(m_gemmKey, "GEMM key is not initialized");
    InnerProductKey key = *m_gemmKey;
    key.src_md = dnnl::memory::desc({static_cast<dnnl::memory::dim>(rows), key.src_md.get_dims()[1]},
                                    key.src_md.get_data_type(),
                                    dnnl::memory::format_tag::ab);
    key.single_thread = single_thread;
    const auto& eng = m_context->getEngine();
    const auto threadPool = m_context->getThreadPool();
    InnerProductPtr gemm;
    std::tie(gemm, std::ignore) =
        m_context->getRuntimeCache()->getOrCreate(key, [&eng, &threadPool](const InnerProductKey& k) {
            return std::make_shared<InnerProduct>(eng, threadPool, k);
        });
    return gemm;
}

void GatherMatmulDnnlExecutor::execute(const MemoryArgs& memory) {
    const auto& cpu_parallel = m_context->getCpuParallel();
    const auto& srcMem = memory.at(ARG_SRC);
    const auto& biasMem = memory.at(ARG_BIAS);
    const auto& indexMem = memory.at(ARG_SRC_1);
    const auto& dstMem = memory.at(ARG_DST);

    const auto& indexShape = indexMem->getStaticDims();
    size_t M = indexShape[0];
    size_t indices_size = indexShape[1];

    auto src_offset = OffsetHelper::createOffsetHelper(srcMem);
    auto dst_offset = OffsetHelper::createOffsetHelper(dstMem);
    auto wei_offset = OffsetHelper::createOffsetHelper(m_weightsMemory);
    auto bias_offset = OffsetHelper::createOffsetHelper(biasMem);
    auto scale_offset = OffsetHelper::createOffsetHelper(m_scalesMemory);
    auto zp_offset = OffsetHelper::createOffsetHelper(m_zpMemory);
    auto index_offset = OffsetHelper::createOffsetHelper(indexMem);

    const size_t gather_axis_size = m_weightsMemory->getStaticDims()[0];

    if (M > 1) {
        std::vector<std::pair<int32_t, int32_t>> gather_idx_map(gather_axis_size * M);
        std::vector<int32_t> elements_per_gather_indx(gather_axis_size, 0);
        for (size_t m = 0; m < M; m++) {
            const auto* gather_ids = static_cast<const int32_t*>(index_offset(m));
            for (size_t i = 0; i < indices_size; i++) {
                int32_t gather_axis_index = gather_ids[i];
                OPENVINO_ASSERT(gather_axis_index >= 0 && static_cast<size_t>(gather_axis_index) < gather_axis_size,
                                "Invalid gather_id ",
                                gather_axis_index,
                                " for m ",
                                m);
                auto& index = elements_per_gather_indx[gather_axis_index];
                gather_idx_map[gather_axis_index * M + index] = {m, i};
                index++;
            }
        }

        // No row threshold: on a 128-expert top-8 int4 MoE the expert-parallel path gave a shorter time to first
        // token at every input length from 2 to 128 rows, on 32 and on 96 cores (1.5 times at 2 rows). A single
        // row (M == 1) runs the GEMV below.
        if (m_expertParallelOk) {
            executeExpertParallel(memory, gather_idx_map, elements_per_gather_indx, M);
            return;
        }

        OPENVINO_ASSERT(m_tmpInpBuffer, "Temporary input/output memory is not created");
        OPENVINO_ASSERT(m_tmpInputDesc, "Temporary input memory desc is not created");
        OPENVINO_ASSERT(m_tmpOutputDesc, "Temporary output memory desc is not created");

        const auto element_size = m_tmpInputDesc->getPrecision().size();
        const auto K_size = m_tmpInputDesc->getShape().getStaticDims()[1];
        const auto M_size = m_tmpInputDesc->getShape().getStaticDims()[0];
        const auto N_size = dstMem->getStaticDims()[2];

        auto* input_ptr = m_tmpInpBuffer->getDataAs<uint8_t>();
        auto* output_ptr = input_ptr + rnd_up(m_tmpInputDesc->getCurrentMemSize(), 64);

        Memory tmpInput(m_context->getEngine(), m_tmpInputDesc, m_tmpInpBuffer->getMemoryBlock());
        Memory tmpOutput(m_context->getEngine(), m_tmpOutputDesc, output_ptr);

        auto tmp_input_offset = OffsetHelper::createOffsetHelper(tmpInput);
        auto tmp_dst_offset = OffsetHelper::createOffsetHelper(tmpOutput);

        // Gather indices with the same rounded row count share one GEMM within this call. The runtime cache
        // keeps the primitives across calls.
        std::unordered_map<size_t, InnerProductPtr> gemmBySize;
        // On AMX a GEMM is cheap even for a few rows; elsewhere experts with fewer rows run per-row GEMV.
        const size_t min_rows_for_gemm = m_bf16AmxMode ? 1 : 4;
        for (size_t gather_axis_index = 0; gather_axis_index < gather_axis_size; gather_axis_index++) {
            const size_t num_valid_rows = elements_per_gather_indx[gather_axis_index];
            if (0 == num_valid_rows) {
                continue;
            }
            OPENVINO_ASSERT(num_valid_rows <= M_size,
                            "GatherMatmul: a gather index received more rows than the input has");

            // Few rows: per-row GEMV is cheaper than gathering, padding, and scattering them for a GEMM.
            // A decode step of N requests is M = N, so most experts get one or two rows.
            if (num_valid_rows < min_rows_for_gemm) {
                for (size_t m = 0; m < num_valid_rows; ++m) {
                    const auto row_id = gather_idx_map[gather_axis_index * M + m].first;
                    const auto batch_index = gather_idx_map[gather_axis_index * M + m].second;
                    m_gemvImpl->exec(src_offset(batch_index, row_id),
                                     dst_offset(batch_index, row_id),
                                     wei_offset(gather_axis_index),
                                     bias_offset(gather_axis_index),
                                     scale_offset(gather_axis_index),
                                     zp_offset(gather_axis_index));
                }
                continue;
            }

            // Size the GEMM to the rows routed to this gather index (rounded up by normalizeM), not to
            // the full input length: with top-k routing most rows of a full-length GEMM are zero padding.
            const size_t gemm_rows = normalizeM(num_valid_rows);
            auto& gemmImpl = gemmBySize[gemm_rows];
            if (!gemmImpl) {
                gemmImpl = createGemm(gemm_rows);
            }

            cpu_parallel->parallel_for(gemm_rows, [&](size_t m) {
                auto* dst_row = tmp_input_offset(m);
                if (m < num_valid_rows) {
                    const auto row_id = gather_idx_map[gather_axis_index * M + m].first;
                    const auto batch_index = gather_idx_map[gather_axis_index * M + m].second;
                    const auto* src_data = src_offset(batch_index, row_id);
                    std::memcpy(dst_row, src_data, K_size * element_size);
                } else {
                    std::memset(dst_row, 0, K_size * element_size);
                }
            });

            auto* src = tmp_input_offset.get_base();
            auto* dst = tmp_dst_offset.get_base();
            auto* wei = wei_offset(gather_axis_index);
            auto* bias = bias_offset(gather_axis_index);
            auto* scale = scale_offset(gather_axis_index);
            auto* zp = zp_offset(gather_axis_index);
            gemmImpl->exec(src, dst, wei, bias, scale, zp);

            cpu_parallel->parallel_for(num_valid_rows, [&](size_t m) {
                const auto* src_row = tmp_dst_offset(m);
                const auto row_id = gather_idx_map[gather_axis_index * M + m].first;
                const auto batch_index = gather_idx_map[gather_axis_index * M + m].second;
                auto* dst_row = dst_offset(batch_index, row_id);
                std::memcpy(dst_row, src_row, N_size * element_size);
            });
        }
    } else {
        OPENVINO_ASSERT(m_gemvImpl, "GEMV implementation is not created");

        constexpr size_t m = 0;
        auto* gather_ids = static_cast<int32_t*>(index_offset(m));
        for (size_t i = 0; i < indices_size; i++) {
            int32_t gather_axis_index = gather_ids[i];
            OPENVINO_ASSERT(gather_axis_index >= 0 && static_cast<size_t>(gather_axis_index) < gather_axis_size,
                            "Invalid gather_id ",
                            gather_axis_index,
                            " for i ",
                            i);
            auto* src = src_offset(i, m);
            auto* dst = dst_offset(i, m);
            auto* wei = wei_offset(gather_axis_index);
            auto* bias = bias_offset(gather_axis_index);
            auto* scale = scale_offset(gather_axis_index);
            auto* zp = zp_offset(gather_axis_index);
            m_gemvImpl->exec(src, dst, wei, bias, scale, zp);
        }
    }
}

void GatherMatmulDnnlExecutor::executeExpertParallel(const MemoryArgs& memory,
                                                     const std::vector<std::pair<int32_t, int32_t>>& gather_idx_map,
                                                     const std::vector<int32_t>& rows_per_gather_index,
                                                     size_t M) {
    const auto& srcMem = memory.at(ARG_SRC);
    const auto& dstMem = memory.at(ARG_DST);
    const auto src_offset = OffsetHelper::createOffsetHelper(srcMem);
    const auto dst_offset = OffsetHelper::createOffsetHelper(dstMem);
    const auto wei_offset = OffsetHelper::createOffsetHelper(m_weightsMemory);
    const auto bias_offset = OffsetHelper::createOffsetHelper(memory.at(ARG_BIAS));
    const auto scale_offset = OffsetHelper::createOffsetHelper(m_scalesMemory);
    const auto zp_offset = OffsetHelper::createOffsetHelper(m_zpMemory);

    const size_t element_size = srcMem->getDesc().getPrecision().size();
    const size_t K_size = srcMem->getStaticDims().back();
    const size_t N_size = dstMem->getStaticDims().back();

    // A task is up to kExpertRowsPerTask consecutive rows routed to one gather index.
    struct Task {
        size_t gather_index;
        size_t start;
        size_t rows;
    };
    std::vector<Task> tasks;
    for (size_t gather_index = 0; gather_index < rows_per_gather_index.size(); gather_index++) {
        const auto rows = static_cast<size_t>(rows_per_gather_index[gather_index]);
        for (size_t start = 0; start < rows; start += kExpertRowsPerTask) {
            tasks.push_back({gather_index, start, std::min(kExpertRowsPerTask, rows - start)});
        }
    }
    // Largest tasks first, so the tasks that start last are short and the threads finish close together.
    std::stable_sort(tasks.begin(), tasks.end(), [](const Task& a, const Task& b) {
        return a.rows > b.rows;
    });

    OPENVINO_ASSERT(m_tmpInpBuffer && m_tmpInpBuffer->getSize() >= m_epSlots * m_epSlotBytes,
                    "GatherMatmul: the expert-parallel work areas are not allocated");
    auto* const regions = m_tmpInpBuffer->getDataAs<uint8_t>();

    std::atomic<size_t> next_task{0};
    m_context->getCpuParallel()->parallel_for(m_epSlots, [&](size_t slot) {
        uint8_t* const input = regions + slot * m_epSlotBytes;
        uint8_t* const output = input + m_epInBytes;
        uint8_t* const scratchpad = output + m_epOutBytes;
        for (size_t t = next_task.fetch_add(1); t < tasks.size(); t = next_task.fetch_add(1)) {
            const auto& task = tasks[t];
            const size_t height = normalizeM(task.rows);
            const auto& gemm = m_epGemms[height / kTaskHeightStep - 1];
            for (size_t m = 0; m < height; m++) {
                auto* dst_row = input + m * K_size * element_size;
                if (m < task.rows) {
                    const auto& [row_id, batch_index] = gather_idx_map[task.gather_index * M + task.start + m];
                    std::memcpy(dst_row, src_offset(batch_index, row_id), K_size * element_size);
                } else {
                    std::memset(dst_row, 0, K_size * element_size);
                }
            }
            gemm->exec_single_thread(input,
                                     output,
                                     wei_offset(task.gather_index),
                                     bias_offset(task.gather_index),
                                     scale_offset(task.gather_index),
                                     zp_offset(task.gather_index),
                                     scratchpad);
            for (size_t m = 0; m < task.rows; m++) {
                const auto& [row_id, batch_index] = gather_idx_map[task.gather_index * M + task.start + m];
                std::memcpy(dst_offset(batch_index, row_id), output + m * N_size * element_size, N_size * element_size);
            }
        }
    });
}

impl_desc_type GatherMatmulDnnlExecutor::implType() const {
    return m_implType;
}

}  // namespace ov::intel_cpu
