// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "random_uniform.hpp"

#include "openvino/core/parallel.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/random_uniform.hpp"

namespace ov::intel_cpu::node {

// Following const values are taken from the original paper:
// https://www.thesalmons.org/john/random123/papers/random123sc11.pdf
constexpr uint32_t CRUSH_RESISTANCE_CONST_LOWER_VALUE = 0x9E3779B9;
constexpr uint32_t CRUSH_RESISTANCE_CONST_UPPER_VALUE = 0xBB67AE85;
constexpr uint64_t STATISTIC_MAXIMIZING_MULTIPLIER_N = 0xD2511F53;
constexpr uint64_t STATISTIC_MAXIMIZING_MULTIPLIER_COUNTER = 0xCD9E8D57;

// Following const values are taken from the original paper (used by PyTorch):
// https://dl.acm.org/doi/pdf/10.1145/272991.272995
constexpr int32_t MERSENNE_STATE_N = 624;
constexpr int32_t MERSENNE_STATE_M = 397;

bool RandomUniform::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                         std::string& errorMessage) noexcept {
    try {
        if (op->get_type_info() != op::v8::RandomUniform::get_type_info_static()) {
            errorMessage = "Only RandomUniform operation from the opset8 is supported by the CPU plugin.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

RandomUniform::RandomUniform(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    // RandomUniform should generate new sequence each run even if all inputs are constants. So that method
    // Node::IsConstant() doesn't return 'True' for RandomUniform with all constant inputs and the node generates new
    // values for each inference, we set 'StrictNoConst' value for 'ConstantType' in ctor.
    constant = ConstantType::StrictNoConst;

    auto rnd_op = as_type_ptr<op::v8::RandomUniform>(op);
    m_global_seed = rnd_op->get_global_seed();
    m_op_seed = rnd_op->get_op_seed();
    m_output_prc = op->get_output_element_type(0);

    const auto alignment = rnd_op->get_alignment();
    switch (alignment) {
    case ov::op::PhiloxAlignment::TENSORFLOW:
        m_algo = PHILOX;
        break;
    case ov::op::PhiloxAlignment::PYTORCH:
        m_algo = MERSENNE_TWISTER;
        break;
    default:
        THROW_CPU_NODE_ERR("Alignment of RandomUniform ", alignment, " is not supported by the CPU plugin.");
    }

    for (size_t i = 0lu; i < op->get_input_size(); i++) {
        if (is_type<op::v0::Constant>(op->get_input_node_ptr(i))) {
            m_const_inputs[i] = true;
        }
    }

    if (m_algo == STL) {
        m_generator = std::default_random_engine{static_cast<uint32_t>(m_op_seed)};
    }
}

void RandomUniform::getSupportedDescriptors() {
    if (getParentEdges().size() != 3) {
        THROW_CPU_NODE_ERR("has incorrect number of input edges.");
    }
    if (getChildEdges().empty()) {
        THROW_CPU_NODE_ERR("has incorrect number of output edges.");
    }
}

void RandomUniform::initSupportedPrimitiveDescriptors() {
    auto shape_prc = getOriginalInputPrecisionAtPort(SHAPE);
    if (!one_of(shape_prc, ov::element::i32, ov::element::i64)) {
        shape_prc = ov::element::i32;
    }

    auto out_prc = getOriginalOutputPrecisionAtPort(0);
    if (out_prc.is_real()) {
        if (one_of(m_algo, PHILOX, MERSENNE_TWISTER) &&
            !one_of(out_prc, ov::element::f32, ov::element::f16, ov::element::bf16)) {
            out_prc = ov::element::f32;
        }

        if (one_of(m_algo, STL) && !one_of(out_prc, ov::element::f32)) {
            out_prc = ov::element::f32;
        }
    } else {
        if (!one_of(out_prc, ov::element::i32, ov::element::i64)) {
            out_prc = ov::element::i32;
        }
    }

    m_output_prc = out_prc;

    addSupportedPrimDesc({{LayoutType::ncsp, shape_prc, m_const_inputs[SHAPE]},
                          {LayoutType::ncsp, out_prc, m_const_inputs[MIN_VAL]},
                          {LayoutType::ncsp, out_prc, m_const_inputs[MAX_VAL]}},
                         {{LayoutType::ncsp, out_prc}},
                         ref_any);
}

void RandomUniform::createPrimitive() {
    if (m_const_inputs[MIN_VAL]) {
        initEdgeValues(m_min_val, getSrcDataAtPort(MIN_VAL), m_output_prc);
    }
    if (m_const_inputs[MAX_VAL]) {
        initEdgeValues(m_max_val, getSrcDataAtPort(MAX_VAL), m_output_prc);
        evalRange();
    }

    prepareGeneratorKernel();

    if (m_const_inputs[SHAPE]) {
        Node::createPrimitive();
    }
}

bool RandomUniform::needPrepareParams() const {
    if (m_out_shape != getDstMemoryAtPort(0)->getShape().getStaticDims()) {
        return true;
    }
    return false;
}

void RandomUniform::prepareParams() {
    m_out_shape = getDstMemoryAtPort(0)->getShape().getStaticDims();
    m_output_elements_count = std::accumulate(m_out_shape.begin(), m_out_shape.end(), 1lu, std::multiplies<>());

    if (m_algo == PHILOX) {
        m_skip_count = m_output_elements_count * SKIP_CONST;
        preparePhiloxParams();
    } else if (m_algo == MERSENNE_TWISTER) {
        prepareMersenneTwisterParams();
    }
}

void RandomUniform::execute(const dnnl::stream& strm) {
    if (!m_const_inputs[MIN_VAL]) {
        initEdgeValues(m_min_val, getSrcDataAtPort(MIN_VAL), m_output_prc);
        if (m_const_inputs[MAX_VAL]) {
            evalRange();
        }
    }
    if (!m_const_inputs[MAX_VAL]) {
        initEdgeValues(m_max_val, getSrcDataAtPort(MAX_VAL), m_output_prc);
        evalRange();
    }

    auto data = getDstDataAtPort(0);

    if (m_algo == PHILOX) {
        m_state = computePhilox(data, m_output_elements_count, m_state);
    } else if (m_algo == MERSENNE_TWISTER) {
        computeMersenneTwister(data, m_output_elements_count);
    } else if (m_algo == STL) {
        computeStl(data, m_output_elements_count);
    } else {
        THROW_CPU_NODE_ERR("does not support the selected algorithm.");
    }
}

void RandomUniform::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

std::string RandomUniform::getPrimitiveDescriptorType() const {
    auto selectedPrimitiveDesc = getSelectedPrimitiveDescriptor();

    impl_desc_type type = impl_desc_type::undef;
    if (selectedPrimitiveDesc) {
        type = selectedPrimitiveDesc->getImplementationType();
    }

    std::string str_type;

    auto add_type = [&](const std::string& t) {
        if (!str_type.empty() && t.c_str()[0] != '_') {
            str_type += "_";
        }
        str_type += t;
    };

#define SEARCH_TYPE(_type)                                       \
    if ((type & impl_desc_type::_type) == impl_desc_type::_type) \
    add_type(#_type)

    SEARCH_TYPE(undef);
    SEARCH_TYPE(jit);
    SEARCH_TYPE(ref);

    SEARCH_TYPE(avx512);
    SEARCH_TYPE(avx2);
    SEARCH_TYPE(sse42);
    SEARCH_TYPE(any);

#undef SEARCH_TYPE

    if (type == impl_desc_type::unknown) {
        str_type = "unknown";
    } else if (str_type.empty()) {
        str_type = "undef";
    }

    if (selectedPrimitiveDesc) {
        if (selectedPrimitiveDesc->getConfig().outConfs[0].getMemDesc()->getPrecision() != ov::element::u8) {
            str_type +=
                "_" + static_cast<std::string>(
                          selectedPrimitiveDesc->getConfig().outConfs[0].getMemDesc()->getPrecision().get_type_name());
        } else {
            str_type += "_I8";
        }
    }

    return str_type;
}

bool RandomUniform::needShapeInfer() const {
    return !m_const_inputs[SHAPE];
}

bool RandomUniform::neverExecute() const {
    return getSelectedPrimitiveDescriptor()->hasZeroInputDimsAtPort(SHAPE);
}

bool RandomUniform::isExecutable() const {
    return !isInputTensorAtPortEmpty(SHAPE);
}

bool RandomUniform::created() const {
    return getType() == Type::RandomUniform;
}

////////////////////////////////////////////////

void RandomUniform::evalRange() {
#define EL_CASE(E)                                 \
    case element::E:                               \
        m_range_val.E = m_max_val.E - m_min_val.E; \
        break;

    switch (m_output_prc) {
        EL_CASE(f64)
        EL_CASE(f32)
        EL_CASE(f16)
        EL_CASE(bf16)
        EL_CASE(i64)
        EL_CASE(i32)
    default:
        THROW_CPU_NODE_ERR("has unsupported output precision: ", m_output_prc);
    }

#undef EL_CASE
}

void RandomUniform::initEdgeValues(OutputType& dst, const void* src, const element::Type& output_type) {
#define EL_CASE(E)                                                                          \
    case element::E:                                                                        \
        dst.E = *reinterpret_cast<const element_type_traits<element::E>::value_type*>(src); \
        break;

    switch (output_type) {
        EL_CASE(f64)
        EL_CASE(f32)
        EL_CASE(f16)
        EL_CASE(bf16)
        EL_CASE(i64)
        EL_CASE(i32)
    default:
        THROW_CPU_NODE_ERR("has unsupported output precision: ", output_type);
    }

#undef EL_CASE
}

void RandomUniform::preparePhiloxParams() {
    if (m_output_elements_count < PHILOX_PARALLEL_EXECUTION_THRESHOLD) {
        m_threads_num = 1;
    } else {
        m_threads_num = parallel_get_max_threads();
    }
    m_philox_thread_params.resize(m_threads_num);

    parallel_nt(m_threads_num, [&](const int ithr, const int nthr) {
        auto& params = m_philox_thread_params[ithr];
        uint64_t start = 0lu, end = 0lu;

        if (m_jit_kernel) {
#if defined(OPENVINO_ARCH_X86_64)
            const auto block_size = (m_jit_kernel->getVectorLen() / m_output_prc.size()) * 2;
            const auto blocks_num = (m_output_elements_count + block_size - 1) / block_size;
            const auto blocks_per_thr = (blocks_num + nthr - 1) / nthr;

            start = ithr * blocks_per_thr * block_size;
            end = (ithr + 1) * blocks_per_thr * block_size;
#endif  // OPENVINO_ARCH_X86_64
        } else {
            const auto groups_num = (m_output_elements_count + PHILOX_GROUP_SIZE - 1) / PHILOX_GROUP_SIZE;
            const auto groups_per_thr = (groups_num + nthr - 1) / nthr;

            start = ithr * groups_per_thr * PHILOX_GROUP_SIZE;
            end = (ithr + 1) * groups_per_thr * PHILOX_GROUP_SIZE;

            params.step = m_output_prc.size() > 4 ? 2 : 4;
        }

        if (end > m_output_elements_count) {
            end = m_output_elements_count;
        }
        if (start > end) {
            start = end;
        }
        params.work_amount = end - start;
        params.n_shift = start / PHILOX_GROUP_SIZE;
        params.dst_shift = start * m_output_prc.size();
    });
}

void RandomUniform::prepareMersenneTwisterParams() {
    m_threads_num = parallel_get_max_threads();

    if (m_jit_kernel) {
#if defined(OPENVINO_ARCH_X86_64)
        // m_jit_kernel->getVectorLen() either 64, 32 or 16 (bytes) for Zmm, Ymm, Xmm respectively
        m_uint_storage_capacity_per_thread = m_jit_kernel->getVectorLen() / sizeof(uint32_t);
        const auto maximum_jit_threads = MERSENNE_STATE_N / m_uint_storage_capacity_per_thread;
        m_threads_num = std::max(std::min(m_threads_num, maximum_jit_threads), 1);
#endif  // OPENVINO_ARCH_X86_64
    } else {
        // Each thread processes a pair of uints, generating either 1 or 2 outputs
        m_uint_storage_capacity_per_thread = 2;
        const auto maximum_threads = MERSENNE_STATE_N / m_uint_storage_capacity_per_thread;
        m_threads_num = std::max(std::min(m_threads_num, maximum_threads), 1);
    }

    m_mersenne_twister_thread_params.resize(m_threads_num);
    m_mersenne_twister_optimization_enabled =
        !(m_output_prc == element::i64 && (m_max_val.i64 > std::numeric_limits<uint32_t>::max() ||
                                           m_min_val.i64 > std::numeric_limits<uint32_t>::max()));

    const auto thread_offset = static_cast<float>(MERSENNE_STATE_N) / static_cast<float>(m_threads_num) /
                               static_cast<float>(m_uint_storage_capacity_per_thread);

    const auto byte_offset = m_output_prc.size() / (m_mersenne_twister_optimization_enabled ? 1 : 2);

    parallel_nt(m_threads_num, [&](int ithr, int nthr) {
        auto& params = m_mersenne_twister_thread_params[ithr];

        auto approx_start = thread_offset * static_cast<float>(ithr);
        auto approx_end = thread_offset * (static_cast<float>(ithr + 1));

        auto state_start = static_cast<uint64_t>(std::floor(approx_start) * m_uint_storage_capacity_per_thread);
        auto state_end = static_cast<uint64_t>(std::floor(approx_end) * m_uint_storage_capacity_per_thread);

        // Rounding failsafes
        if (ithr == 0) {
            state_start = 0;
        } else if (ithr + 1 == m_threads_num) {
            state_end = MERSENNE_STATE_N;
        }

        auto state_accesses = std::ceil(static_cast<float>(state_end - state_start) /
                                        static_cast<float>(m_uint_storage_capacity_per_thread));

        // Destination index is computed in bytes, therefore the state index
        // has to be divided by the byte size of dtype.
        // in addition, when optimization is off, 2 values are consumed to create
        // one output value, so the state index has to be divided by 2
        auto destination_start = state_start * byte_offset;

        params.src_start_idx = state_start;
        params.dst_start_idx = destination_start;
        params.state_accesses_count = state_accesses;
    });
}

void RandomUniform::prepareGeneratorKernel() {
#if defined(OPENVINO_ARCH_X86_64)
    if (m_algo == PHILOX) {
        kernel::random_uniform::PhiloxGeneratorCompileParams jcp;
        jcp.out_data_type = m_output_prc;

        m_jit_kernel = kernel::JitKernel<kernel::random_uniform::PhiloxGeneratorCompileParams,
                                         kernel::random_uniform::PhiloxGeneratorCallArgs>::
            createInstance<kernel::random_uniform::PhiloxGenerator>(jcp);
    } else if (m_algo == MERSENNE_TWISTER) {
        kernel::random_uniform::MersenneTwisterGeneratorCompileParams jcp;
        jcp.out_data_type = m_output_prc;
        jcp.optimized = m_mersenne_twister_optimization_enabled;

        m_jit_kernel = kernel::JitKernel<kernel::random_uniform::MersenneTwisterGeneratorCompileParams,
                                         kernel::random_uniform::MersenneTwisterGeneratorCallArgs>::
            createInstance<kernel::random_uniform::MersenneTwisterGenerator>(jcp);
    }

    if (m_jit_kernel) {
        if (auto selected_pd = getSelectedPrimitiveDescriptor()) {
            using namespace dnnl::impl::cpu;
            if (m_jit_kernel->getIsa() == x64::avx512_core) {
                selected_pd->setImplementationType(jit_avx512);
            } else if (m_jit_kernel->getIsa() == x64::avx2) {
                selected_pd->setImplementationType(jit_avx2);
            } else if (m_jit_kernel->getIsa() == x64::sse41) {
                selected_pd->setImplementationType(jit_sse42);
            }
        }
    }
#endif  // OPENVINO_ARCH_X86_64
}

////////////// PHILOX algo ///////////////

namespace {

inline void calculateRound(const uint32_t* key, uint32_t* counter, uint32_t* n) {
    uint64_t prod_0 = STATISTIC_MAXIMIZING_MULTIPLIER_N * n[0];
    uint64_t prod_1 = STATISTIC_MAXIMIZING_MULTIPLIER_COUNTER * counter[0];
    n[0] = static_cast<uint32_t>(prod_1 >> 32) ^ n[1] ^ key[0];
    n[1] = static_cast<uint32_t>(prod_1);
    counter[0] = static_cast<uint32_t>(prod_0 >> 32) ^ counter[1] ^ key[1];
    counter[1] = static_cast<uint32_t>(prod_0);
}

inline void raiseKey(uint32_t* key) {
    key[0] += CRUSH_RESISTANCE_CONST_LOWER_VALUE;
    key[1] += CRUSH_RESISTANCE_CONST_UPPER_VALUE;
}

inline void runPhilox(uint64_t key, uint64_t counter, uint64_t n, uint32_t* res) {
    auto* key_32 = reinterpret_cast<uint32_t*>(&key);
    auto* counter_32 = reinterpret_cast<uint32_t*>(&counter);
    auto* n_32 = reinterpret_cast<uint32_t*>(&n);

    // Loop unwarping for better performance
    calculateRound(key_32, counter_32, n_32);
    raiseKey(key_32);
    calculateRound(key_32, counter_32, n_32);
    raiseKey(key_32);
    calculateRound(key_32, counter_32, n_32);
    raiseKey(key_32);
    calculateRound(key_32, counter_32, n_32);
    raiseKey(key_32);
    calculateRound(key_32, counter_32, n_32);
    raiseKey(key_32);
    calculateRound(key_32, counter_32, n_32);
    raiseKey(key_32);
    calculateRound(key_32, counter_32, n_32);
    raiseKey(key_32);
    calculateRound(key_32, counter_32, n_32);
    raiseKey(key_32);
    calculateRound(key_32, counter_32, n_32);
    raiseKey(key_32);
    calculateRound(key_32, counter_32, n_32);

    res[0] = n_32[0];
    res[1] = n_32[1];
    res[2] = counter_32[0];
    res[3] = counter_32[1];
}

inline void convertToOutputTypePhilox(const uint32_t* in, float min, float range, float* out, size_t el_to_copy) {
    RandomUniform::OutputType out_val;

    for (size_t i = 0lu; i < el_to_copy; i++) {
        out_val.u32 = 0x3f800000 | (in[i] & 0x7fffffu);
        out[i] = (out_val.f32 - 1.f) * range + min;
    }
}

inline void convertToOutputTypePhilox(const uint32_t* in, float16 min, float16 range, float16* out, size_t el_to_copy) {
    RandomUniform::OutputType out_val;

    for (size_t i = 0lu; i < el_to_copy; i++) {
        auto x_uint16 = static_cast<uint16_t>(in[i]);
        out_val.u16 = 0x3c00 | (x_uint16 & 0x03ffu);
        out[i] = (out_val.f16 - static_cast<float16>(1)) * range + min;
    }
}

inline void convertToOutputTypePhilox(const uint32_t* in,
                                      bfloat16 min,
                                      bfloat16 range,
                                      bfloat16* out,
                                      size_t el_to_copy) {
    RandomUniform::OutputType out_val;

    for (size_t i = 0lu; i < el_to_copy; i++) {
        auto x_uint16 = static_cast<uint16_t>(in[i]);
        out_val.u16 = 0x3f80 | (x_uint16 & 0x7fu);
        out[i] = (out_val.bf16 - static_cast<bfloat16>(1)) * range + min;
    }
}

inline void convertToOutputTypePhilox(const uint32_t* in, int32_t min, int32_t range, int32_t* out, size_t el_to_copy) {
    for (size_t i = 0lu; i < el_to_copy; i++) {
        out[i] = static_cast<int32_t>(in[i] % range + min);
    }
}

inline void convertToOutputTypePhilox(const uint32_t* in, int64_t min, int64_t range, int64_t* out, size_t el_to_copy) {
    for (size_t i = 0lu; i < el_to_copy; i++) {
        out[i] = static_cast<int64_t>(((static_cast<uint64_t>(in[i * 2]) << 32) + in[i * 2 + 1]) % range + min);
    }
}

}  // namespace

std::pair<uint64_t, uint64_t> RandomUniform::computePhilox(void* out,
                                                           size_t output_elements_count,
                                                           const std::pair<uint64_t, uint64_t>& prev_state) {
    // When both seed values are equal to zero RandomUniform should generate non-deterministic sequence.
    if (m_global_seed == 0lu && m_op_seed == 0lu) {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        m_global_seed = std::rand();
    }

    uint64_t n_state = prev_state.first;
    uint64_t counter_state = prev_state.second;
    uint64_t counter = counter_state > 0 ? counter_state : m_op_seed;

    auto out_u8 = reinterpret_cast<uint8_t*>(out);

    if (m_jit_kernel) {
#if defined(OPENVINO_ARCH_X86_64)
        parallel_nt(m_threads_num, [&](const int ithr, const int nthr) {
            auto& params = m_philox_thread_params[ithr];
            if (params.work_amount == 0lu) {
                return;
            }
            auto n = n_state + params.n_shift;

            kernel::random_uniform::PhiloxGeneratorCallArgs args;

            args.dst_ptr = (out_u8 + params.dst_shift);
            args.key_ptr = &m_global_seed;
            args.counter_ptr = &counter;
            args.n_ptr = &n;
            args.min_ptr = &m_min_val;
            args.range_ptr = &m_range_val;
            args.work_amount = params.work_amount;

            (*m_jit_kernel)(&args);
        });
#endif  // OPENVINO_ARCH_X86_64
    } else {
        auto threadBody = [&](const int ithr, const int nthr) {
            auto& params = m_philox_thread_params[ithr];
            if (params.work_amount == 0lu) {
                return;
            }
            auto n = n_state + params.n_shift;
            auto out_cur = out_u8 + params.dst_shift;
            auto work_rest = static_cast<int64_t>(params.work_amount);
            uint32_t res[4];

#define EXEC_CASE(P)                                                                          \
    case element::P: {                                                                        \
        auto out_t = reinterpret_cast<element_type_traits<element::P>::value_type*>(out_cur); \
        for (; work_rest > 0l; work_rest -= params.step, out_t += params.step) {              \
            runPhilox(m_global_seed, counter, n, res);                                        \
            auto el_to_copy = std::min(params.step, static_cast<uint64_t>(work_rest));        \
            convertToOutputTypePhilox(res, m_min_val.P, m_range_val.P, out_t, el_to_copy);    \
            if (++n == 0) {                                                                   \
                counter++;                                                                    \
            }                                                                                 \
        }                                                                                     \
    } break;

            switch (m_output_prc) {
                EXEC_CASE(f32)
                EXEC_CASE(f16)
                EXEC_CASE(bf16)
                EXEC_CASE(i32)
                EXEC_CASE(i64)
            default:
                THROW_CPU_NODE_ERR("Unsupported type of RandomUniform: ", m_output_prc.to_string());
            }

#undef EXEC_CASE
        };

        parallel_nt(m_threads_num, threadBody);
    }

    // Calculate counter values for next RandomUniform run.
    n_state += m_skip_count;
    if (n_state < m_skip_count) {
        counter_state++;
    }

    return {n_state, counter_state};
}

////////////// MERSENNE algo ///////////////

namespace {

uint32_t twist(uint32_t u, uint32_t v) {
    return (((u & 0x80000000) | (v & 0x7fffffff)) >> 1) ^ (v & 1 ? 0x9908b0df : 0);
}

inline void initial_mersenne_state(uint32_t* mersenne_state_ptr, uint64_t global_seed) {
    mersenne_state_ptr[0] = global_seed & 0xffffffff;
    for (uint32_t j = 1; j < MERSENNE_STATE_N; ++j) {
        mersenne_state_ptr[j] = (1812433253 * (mersenne_state_ptr[j - 1] ^ (mersenne_state_ptr[j - 1] >> 30)) + j);
    }
}

inline void next_mersenne_state(uint32_t* mersenne_state_ptr) {
    auto* current_state_ptr = mersenne_state_ptr;
    for (int j = MERSENNE_STATE_N - MERSENNE_STATE_M + 1; --j; current_state_ptr++) {
        *current_state_ptr = current_state_ptr[MERSENNE_STATE_M] ^ twist(current_state_ptr[0], current_state_ptr[1]);
    }

    for (int j = MERSENNE_STATE_M; --j; current_state_ptr++) {
        *current_state_ptr =
            current_state_ptr[MERSENNE_STATE_M - MERSENNE_STATE_N] ^ twist(current_state_ptr[0], current_state_ptr[1]);
    }

    *current_state_ptr =
        current_state_ptr[MERSENNE_STATE_M - MERSENNE_STATE_N] ^ twist(current_state_ptr[0], mersenne_state_ptr[0]);
}

void runMersenneTwister(uint32_t& random_nr_1, uint32_t& random_nr_2) {
    random_nr_1 ^= (random_nr_1 >> 11);
    random_nr_1 ^= (random_nr_1 << 7) & 0x9d2c5680;
    random_nr_1 ^= (random_nr_1 << 15) & 0xefc60000;
    random_nr_1 ^= (random_nr_1 >> 18);

    random_nr_2 ^= (random_nr_2 >> 11);
    random_nr_2 ^= (random_nr_2 << 7) & 0x9d2c5680;
    random_nr_2 ^= (random_nr_2 << 15) & 0xefc60000;
    random_nr_2 ^= (random_nr_2 >> 18);
}

inline void convertToOutputTypeMersenne(const uint32_t in1,
                                        const uint32_t in2,
                                        float min,
                                        float range,
                                        float* out,
                                        int64_t elements_remaining,
                                        bool optimization_enabled) {
    const auto mask = static_cast<uint32_t>((static_cast<uint64_t>(1) << std::numeric_limits<float>::digits) - 1);
    const auto divisor = static_cast<float>(1) / (static_cast<uint64_t>(1) << std::numeric_limits<float>::digits);

    out[0] = static_cast<float>((in1 & mask) * divisor) * range + min;
    if (elements_remaining >= 2l) {
        out[1] = static_cast<float>((in2 & mask) * divisor) * range + min;
    }
}

inline void convertToOutputTypeMersenne(const uint32_t in1,
                                        const uint32_t in2,
                                        float16 min,
                                        float16 range,
                                        float16* out,
                                        int64_t elements_remaining,
                                        bool optimization_enabled) {
    const auto mask = static_cast<uint32_t>((static_cast<uint64_t>(1) << std::numeric_limits<float16>::digits) - 1);
    const auto divisor = static_cast<float>(1) / (static_cast<uint64_t>(1) << std::numeric_limits<float16>::digits);

    out[0] = static_cast<float>((in1 & mask) * divisor) * range + min;
    if (elements_remaining >= 2l) {
        out[1] = static_cast<float>((in2 & mask) * divisor) * range + min;
    }
}

inline void convertToOutputTypeMersenne(const uint32_t in1,
                                        const uint32_t in2,
                                        bfloat16 min,
                                        bfloat16 range,
                                        bfloat16* out,
                                        int64_t elements_remaining,
                                        bool optimization_enabled) {
    const auto mask = static_cast<uint32_t>((1UL << 8) - 1);
    const auto divisor = static_cast<float>(1) / (1UL << 8);

    out[0] = static_cast<float>((in1 & mask) * divisor) * range + min;
    if (elements_remaining >= 2l) {
        out[1] = static_cast<float>((in2 & mask) * divisor) * range + min;
    }
}

inline void convertToOutputTypeMersenne(const uint32_t in1,
                                        const uint32_t in2,
                                        int32_t min,
                                        int32_t range,
                                        int32_t* out,
                                        int64_t elements_remaining,
                                        bool optimization_enabled) {
    out[0] = static_cast<int32_t>(in1 % range + min);
    if (elements_remaining >= 2l) {
        out[1] = static_cast<int32_t>(in2 % range + min);
    }
}

inline void convertToOutputTypeMersenne(const uint32_t in1,
                                        const uint32_t in2,
                                        int64_t min,
                                        int64_t range,
                                        int64_t* out,
                                        int64_t elements_remaining,
                                        bool optimization_enabled) {
    if (optimization_enabled) {
        out[0] = static_cast<int64_t>(in1 % range + min);
        if (elements_remaining >= 2l) {
            out[1] = static_cast<int64_t>(in2 % range + min);
        }
    } else {
        out[0] = static_cast<int64_t>(((static_cast<uint64_t>(in1) << 32) + in2) % range + min);
    }
}
}  // namespace

void RandomUniform::computeMersenneTwister(void* out, size_t output_elements_count) {
    // When both seed values are equal to zero RandomUniform should generate non-deterministic sequence.
    if (m_global_seed == 0lu && m_op_seed == 0lu) {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        m_global_seed = std::rand();
    }

    const auto elements_consumed_per_one_output = m_mersenne_twister_optimization_enabled ? 1 : 2;
    const auto state_regenerations_required =
        static_cast<uint64_t>(std::ceil(static_cast<double>(output_elements_count) /
                                        (static_cast<double>(MERSENNE_STATE_N) / elements_consumed_per_one_output)));
    const auto byte_offset = MERSENNE_STATE_N * m_output_prc.size();

    uint32_t mersenne_state_ptr[MERSENNE_STATE_N];
    auto output_byte_ptr = reinterpret_cast<uint8_t*>(out);
    initial_mersenne_state(mersenne_state_ptr, m_global_seed);

    if (m_jit_kernel) {
#if defined(OPENVINO_ARCH_X86_64)
        for (uint64_t i = 0; i < state_regenerations_required; ++i) {
            next_mersenne_state(mersenne_state_ptr);
            parallel_nt(m_threads_num, [&](const int ithr, const int nthr) {
                kernel::random_uniform::MersenneTwisterGeneratorCallArgs args;
                auto& params = m_mersenne_twister_thread_params[ithr];
                args.min_ptr = &m_min_val;
                args.range_ptr = &m_range_val;
                args.max_output_idx = output_elements_count;
                args.state_accesses_count = params.state_accesses_count;
                args.state_ptr = mersenne_state_ptr + params.src_start_idx;
                args.dst_ptr = output_byte_ptr + params.dst_start_idx + i * byte_offset;
                args.output_idx = params.src_start_idx + i * MERSENNE_STATE_N;
                args.elements_to_generate = static_cast<int64_t>(
                    std::min(static_cast<uint64_t>(m_uint_storage_capacity_per_thread) * args.state_accesses_count,
                             args.max_output_idx - args.output_idx));

                if (args.output_idx >= args.max_output_idx) {
                    return;
                }

                // For loop could not be inside the kernel as I ran out of Reg64s available
                for (uint64_t j = 0; j < args.state_accesses_count; ++j) {
                    (*m_jit_kernel)(&args);

                    args.elements_to_generate =
                        std::max(args.elements_to_generate - static_cast<int64_t>(m_uint_storage_capacity_per_thread),
                                 static_cast<int64_t>(0));
                    args.state_ptr = reinterpret_cast<uint8_t*>(args.state_ptr) +
                                     m_uint_storage_capacity_per_thread * m_output_prc.size();
                    args.dst_ptr = reinterpret_cast<uint8_t*>(args.dst_ptr) +
                                   m_uint_storage_capacity_per_thread * m_output_prc.size();
                    args.output_idx += m_uint_storage_capacity_per_thread;
                }
            });
        }
#endif  // OPENVINO_ARCH_X86_64
    } else {
        const auto elements_generated_per_access = m_mersenne_twister_optimization_enabled
                                                       ? m_uint_storage_capacity_per_thread
                                                       : m_uint_storage_capacity_per_thread / 2;

        for (uint64_t i = 0; i < state_regenerations_required; ++i) {
            next_mersenne_state(mersenne_state_ptr);
            parallel_nt(m_threads_num, [&](const int ithr, const int nthr) {
                auto& params = m_mersenne_twister_thread_params[ithr];
                auto state_ptr = mersenne_state_ptr + params.src_start_idx;
                auto dst_ptr = output_byte_ptr + params.dst_start_idx + i * byte_offset;
                auto output_idx = params.src_start_idx + i * MERSENNE_STATE_N;
                auto max_output_idx = output_elements_count;
                auto state_accesses_count = params.state_accesses_count;
                auto elements_to_generate = static_cast<int64_t>(
                    std::min(static_cast<uint64_t>(m_uint_storage_capacity_per_thread) * state_accesses_count,
                             max_output_idx - output_idx));

                if (output_idx == max_output_idx) {
                    return;
                }

#define EXEC_CASE(P)                                                                                  \
    case element::P: {                                                                                \
        auto dst_dtype_ptr = reinterpret_cast<element_type_traits<element::P>::value_type*>(dst_ptr); \
        for (uint64_t j = 0; j < state_accesses_count; ++j) {                                         \
            if (output_idx >= max_output_idx) {                                                       \
                return;                                                                               \
            }                                                                                         \
                                                                                                      \
            uint32_t random_nr_1 = state_ptr[0], random_nr_2 = state_ptr[1];                          \
            runMersenneTwister(random_nr_1, random_nr_2);                                             \
            convertToOutputTypeMersenne(random_nr_1,                                                  \
                                        random_nr_2,                                                  \
                                        m_min_val.P,                                                  \
                                        m_range_val.P,                                                \
                                        dst_dtype_ptr,                                                \
                                        elements_to_generate,                                         \
                                        m_mersenne_twister_optimization_enabled);                     \
                                                                                                      \
            elements_to_generate -= elements_generated_per_access;                                    \
            state_ptr += m_uint_storage_capacity_per_thread;                                          \
            dst_dtype_ptr += elements_generated_per_access;                                           \
            output_idx += elements_generated_per_access;                                              \
        }                                                                                             \
    } break;

                switch (m_output_prc) {
                    EXEC_CASE(f32)
                    EXEC_CASE(f16)
                    EXEC_CASE(bf16)
                    EXEC_CASE(i32)
                    EXEC_CASE(i64)
                default:
                    THROW_CPU_NODE_ERR("Unsupported type of RandomUniform: ", m_output_prc.to_string());
                }
            });
        }
    }
#undef EXEC_CASE
}

////////////// STL algorithm ///////////////

template <typename T, typename DISTR_TYPE>
void RandomUniform::generateData(DISTR_TYPE distribution, void* out, size_t work_amount) {
    auto dst = reinterpret_cast<T*>(out);
    for (size_t i = 0; i < work_amount; i++) {
        *dst = distribution(m_generator);
        dst++;
    }
}

void RandomUniform::computeStl(void* out, size_t work_amount) {
    switch (m_output_prc) {
    case element::f32: {
        generateData<float, std::uniform_real_distribution<float>>(
            std::uniform_real_distribution<float>{m_min_val.f32, m_max_val.f32},
            out,
            work_amount);
    } break;
    case element::i32: {
        generateData<int32_t, std::uniform_int_distribution<int32_t>>(
            std::uniform_int_distribution<int32_t>{m_min_val.i32, m_max_val.i32},
            out,
            work_amount);
    } break;
    case element::i64: {
        generateData<int64_t, std::uniform_int_distribution<int64_t>>(
            std::uniform_int_distribution<int64_t>{m_min_val.i64, m_max_val.i64},
            out,
            work_amount);
    } break;
    default:
        THROW_CPU_NODE_ERR("has unsupported output type: ", m_output_prc);
    }
}

//////////////////////////////////

}  // namespace ov::intel_cpu::node
