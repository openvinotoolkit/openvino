// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "random_uniform.hpp"

#include "openvino/core/parallel.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/random_uniform.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

bool RandomUniform::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
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
        : Node(op, context, NgraphShapeInferFactory(op, PortMask(0, 1, 2))) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        THROW_CPU_NODE_ERR(errorMessage);
    }

    // RandomUniform should generate new sequence each run even if all inputs are constants. So that method Node::IsConstant()
    // doesn't return 'True' for RandomUniform with all constant inputs and the node generates new values for each inference,
    // we set 'StrictNoConst' value for 'ConstantType' in ctor.
    constant = ConstantType::StrictNoConst;

    auto rnd_op = as_type_ptr<op::v8::RandomUniform>(op);
    m_global_seed = rnd_op->get_global_seed();
    m_op_seed = rnd_op->get_op_seed();
    m_output_prc = op->get_output_element_type(0);

    const auto alignment = rnd_op->get_alignment();
    switch (alignment) {
        case ov::op::PhilloxAlignment::TENSORFLOW:
            m_algo = PHILOX;
            break;
        case ov::op::PhilloxAlignment::PYTORCH:
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
        if (one_of(m_algo, PHILOX, MERSENNE_TWISTER) && !one_of(out_prc, ov::element::f32, ov::element::f16, ov::element::bf16)) {
            out_prc = ov::element::f32;
        }
        
        if (m_algo == STL && !one_of(out_prc, ov::element::f32))) {
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
    m_out_el_num = std::accumulate(m_out_shape.begin(), m_out_shape.end(), 1lu, std::multiplies<Dim>());

    if (m_algo == PHILOX) {
        m_skip_count = m_out_el_num * SKIP_CONST;
        prepareAlgorithmSpecificParams(PHILOX_GROUP_SIZE, PHILOX_PARALLEL_EXECUTION_THRESHOLD);
    } else if (m_algo == MERSENNE_TWISTER) {
        prepareAlgorithmSpecificParams(MERSENNE_TWISTER_GROUP_SIZE, MERSENNE_TWISTER_PARALLEL_EXECUTION_THRESHOLD);
    }
}

void RandomUniform::execute(dnnl::stream strm) {
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
        m_state = computePhilox(data, m_out_el_num, m_state);
    } else if (m_algo == MERSENNE_TWISTER) {
        computeMersenneTwister(data, m_out_el_num);
    } else if (m_algo == STL) {
        computeStl(data, m_out_el_num);
    } else {
        THROW_CPU_NODE_ERR("does not support the selected algorithm.");
    }
}

void RandomUniform::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

std::string RandomUniform::getPrimitiveDescriptorType() const {
    auto selectedPrimitiveDesc = getSelectedPrimitiveDescriptor();

    impl_desc_type type = impl_desc_type::undef;
    if (selectedPrimitiveDesc) {
        type = selectedPrimitiveDesc->getImplementationType();
    }

    std::string str_type;

    auto add_type = [&](std::string t) {
        if (!str_type.empty() && t.c_str()[0] != '_')
            str_type += "_";
        str_type += t;
    };

#define SEARCH_TYPE(_type)                                          \
    if ((type & impl_desc_type::_type) == impl_desc_type::_type)    \
        add_type(#_type)

    SEARCH_TYPE(undef);
    SEARCH_TYPE(jit);
    SEARCH_TYPE(ref);

    SEARCH_TYPE(avx512);
    SEARCH_TYPE(avx2);
    SEARCH_TYPE(sse42);
    SEARCH_TYPE(any);

#undef SEARCH_TYPE

    if (type == impl_desc_type::unknown)
        str_type = "unknown";
    else if (str_type.empty())
        str_type = "undef";

    if (selectedPrimitiveDesc) {
        if (selectedPrimitiveDesc->getConfig().outConfs[0].getMemDesc()->getPrecision() != ov::element::u8) {
            str_type += "_" + std::string(selectedPrimitiveDesc->getConfig().outConfs[0].getMemDesc()->getPrecision().get_type_name());
        } else {
            str_type += "_I8";
        }
    }

    return str_type;
}

bool RandomUniform::needShapeInfer() const {
    return !m_const_inputs[SHAPE];
}

bool RandomUniform::isExecutable() const {
    return !isInputTensorAtPortEmpty(SHAPE);
}

bool RandomUniform::created() const {
    return getType() == Type::RandomUniform;
}

////////////////////////////////////////////////

void RandomUniform::evalRange() {
#define EL_CASE(E) \
    case element::E: \
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
#define EL_CASE(E) \
    case element::E: \
        dst.E = *reinterpret_cast<const element_type_traits<element::E>::value_type *>(src); \
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
    if (m_out_el_num < PHILOX_PARALLEL_EXECUTION_THRESHOLD) {
        m_threads_num = 1;
    } else {
        m_threads_num = parallel_get_max_threads();
    }
    m_thread_params.resize(m_threads_num);

    parallel_nt(m_threads_num, [&](const int ithr, const int nthr) {
        auto& params = m_thread_params[ithr];
        uint64_t start = 0lu, end = 0lu;

        if (m_jit_kernel) {
#if defined(OPENVINO_ARCH_X86_64)
            const auto block_size = (m_jit_kernel->getVectorLen() / m_output_prc.size()) * 2;
            const auto blocks_num = (m_out_el_num + block_size - 1) / block_size;
            const auto blocks_per_thr = (blocks_num + nthr - 1) / nthr;

            start = ithr * blocks_per_thr * block_size;
            end = (ithr + 1) * blocks_per_thr * block_size;
#endif // OPENVINO_ARCH_X86_64
        } else {
            const auto groups_num = (m_out_el_num + PHILOX_GROUP_SIZE - 1) / PHILOX_GROUP_SIZE;
            const auto groups_per_thr = (groups_num + nthr - 1) / nthr;

            start = ithr * groups_per_thr * PHILOX_GROUP_SIZE;
            end = (ithr + 1) * groups_per_thr * PHILOX_GROUP_SIZE;

            params.step = m_output_prc.size() > 4 ? 2 : 4;
        }

        if (end > m_out_el_num) {
            end = m_out_el_num;
        }
        if (start > end) {
            start = end;
        }
        params.work_amount = end - start;
        params.state_shift = start / PHILOX_GROUP_SIZE;
        params.dst_shift = start * m_output_prc.size();
    });
}

void RandomUniform::prepareMersenneTwisterParams() {
    if (m_out_el_num < MERSENNE_TWISTER_PARALLEL_EXECUTION_THRESHOLD) {
        m_threads_num = 1;
    } else {
        auto max_threads = parallel_get_max_threads();
        if (max_threads < MERSENNE_TWISTER_MAXIMUM_THREADS_THRESHOLD) {
            m_threads_num = max_threads
        } else {
            m_threads_num = MERSENNE_TWISTER_MAXIMUM_THREADS_THRESHOLD;
        }
    }
    m_thread_params.resize(m_threads_num);

    if (m_output_prc == element::i64) {
        m_mersenne_twister_optimization_enabled = m_min_val.i64 <= std::numeric_limits<uint32_t>::max() && m_max_val <= std::numeric_limits<uint32_t>::max();
    }

    const auto thread_offset = static_cast<float>(MERSENNE_STATE_N) / static_cast<float>(m_threads_num);
    m_elements_generated = m_output_prc.size() > 4 && !m_mersenne_twister_optimization_enabled ? 2 : 4;

    parallel_nt(m_threads_num, [&](const int ithr, const int nthr) {
        auto& params = m_thread_params[ithr];
        uint64_t start = 0lu, end = 0lu;
        float start_f = 0.f, end_f = 0.f;

        if (m_jit_kernel) {
#if defined(OPENVINO_ARCH_X86_64)
            const auto block_size = (m_jit_kernel->getVectorLen() / m_output_prc.size()) * 2;
            const auto blocks_num = (m_out_el_num + block_size - 1) / block_size;
            const auto blocks_per_thr = (blocks_num + nthr - 1) / nthr;

            start = ithr * blocks_per_thr * block_size;
            end = (ithr + 1) * blocks_per_thr * block_size;
#endif // OPENVINO_ARCH_X86_64
        } else {
            start_f = ithr * thread_offset;
            end_f = (ithr + 1) * thread_offset;

            start = std::floor(start_f);
            end = std::floor(end_f);  
        }

        if (end > m_out_el_num) {
            end = m_out_el_num;
        }
        if (start > end) {
            start = end;
        }
        params.state_shift = start; // idx in mersenne_state
        params.work_amount = end - start; // how many times to generate 4 random numbers in one thread
        params.dst_shift = start * m_output_prc.size();
        params.step = m_elements_generated;
    });
}

void RandomUniform::prepareGeneratorKernel() {
    if (one_of(m_algo, PHILOX, MERSENNE_TWISTER)) {
#if defined(OPENVINO_ARCH_X86_64)

        kernel::random_uniform::GeneratorCompileParams jcp;
        jcp.out_data_type = m_output_prc;

        if (m_algo == PHILOX) {
            m_jit_kernel = kernel::JitKernel<kernel::random_uniform::GeneratorCompileParams,
                                            kernel::random_uniform::PhiloxGeneratorCallArgs>
                                            ::createInstance<kernel::random_uniform::PhiloxGenerator>(jcp);
        } else {
            m_jit_kernel = kernel::JitKernel<kernel::random_uniform::GeneratorCompileParams,
                                            kernel::random_uniform::MersenneTwisterGeneratorCallArgs>
                                            ::createInstance<kernel::random_uniform::MersenneTwisterGenerator>(jcp);
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
#endif // OPENVINO_ARCH_X86_64
    }
}


////////////// PHILOX algo ///////////////

namespace {
// Following const values are taken from the original paper:
// https://www.thesalmons.org/john/random123/papers/random123sc11.pdf
constexpr uint32_t CRUSH_RESISTANCE_CONST_LOWER_VALUE = 0x9E3779B9;
constexpr uint32_t CRUSH_RESISTANCE_CONST_UPPER_VALUE = 0xBB67AE85;
constexpr uint64_t STATISTIC_MAXIMIZING_MULTIPLIER_N = 0xD2511F53;
constexpr uint64_t STATISTIC_MAXIMIZING_MULTIPLIER_COUNTER = 0xCD9E8D57;
constexpr uint64_t ROUNDS_NUMBER = 10llu;

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
    uint32_t* key_32 = reinterpret_cast<uint32_t*>(&key);
    uint32_t* counter_32 = reinterpret_cast<uint32_t*>(&counter);
    uint32_t* n_32 = reinterpret_cast<uint32_t*>(&n);

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

inline void convertToOutputTypePhilox(const uint32_t* in,
                                float min,
                                float range,
                                float* out,
                                size_t el_to_copy) {
    RandomUniform::OutputType out_val;

    for (size_t i = 0lu; i < el_to_copy; i++) {
        out_val.u32 = 0x3f800000 | (in[i] & 0x7fffffu);
        out[i] = (out_val.f32 - 1.f) * range + min;
    }
}

inline void convertToOutputTypePhilox(const uint32_t* in,
                                float16 min,
                                float16 range,
                                float16* out,
                                size_t el_to_copy) {
    RandomUniform::OutputType out_val;

    for (size_t i = 0lu; i < el_to_copy; i++) {
        uint16_t x_uint16 = static_cast<uint16_t>(in[i]);
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
        uint16_t x_uint16 = static_cast<uint16_t>(in[i]);
        out_val.u16 = 0x3f80 | (x_uint16 & 0x7fu);
        out[i] = (out_val.bf16 - static_cast<bfloat16>(1)) * range + min;
    }
}

inline void convertToOutputTypePhilox(const uint32_t* in,
                                int32_t min,
                                int32_t range,
                                int32_t* out,
                                size_t el_to_copy) {
    for (size_t i = 0lu; i < el_to_copy; i++) {
        out[i] = static_cast<int32_t>(in[i] % range + min);
    }
}

inline void convertToOutputTypePhilox(const uint32_t* in,
                                int64_t min,
                                int64_t range,
                                int64_t* out,
                                size_t el_to_copy) {
    for (size_t i = 0lu; i < el_to_copy; i++) {
        out[i] = static_cast<int64_t>(((static_cast<uint64_t>(in[i * 2]) << 32) + in[i * 2 + 1]) % range + min);
    }
}

}  // namespace

std::pair<uint64_t, uint64_t> RandomUniform::computePhilox(void* out, size_t out_el_num, const std::pair<uint64_t, uint64_t>& prev_state) {
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
                auto& params = m_thread_params[ithr];
                if (params.work_amount == 0lu) {
                    return;
                }
                auto n = n_state + params.state_shift;

                kernel::random_uniform::PhiloxGeneratorCallArgs args;

                args.dst_ptr     = (out_u8 + params.dst_shift);
                args.key_ptr     = &m_global_seed;
                args.counter_ptr = &counter;
                args.n_ptr       = &n;
                args.min_ptr     = &m_min_val;
                args.range_ptr   = &m_range_val;
                args.work_amount = params.work_amount;

                (*m_jit_kernel)(&args);
            });
#endif // OPENVINO_ARCH_X86_64
    } else {
        auto threadBody = [&](const int ithr, const int nthr) {
            auto& params = m_thread_params[ithr];
            if (params.work_amount == 0lu) {
                return;
            }
            auto n = n_state + params.state_shift;
            auto out_cur = out_u8 + params.dst_shift;
            auto work_rest = static_cast<int64_t>(params.work_amount);
            uint32_t res[4];

#define EXEC_CASE(P)                                                                                    \
            case element::P: {                                                                          \
                auto out_t = reinterpret_cast<element_type_traits<element::P>::value_type *>(out_cur);  \
                for (; work_rest > 0l; work_rest -= params.step, out_t += params.step) {                          \
                    runPhilox(m_global_seed, counter, n, res);                                          \
                    auto el_to_copy = std::min(params.step, static_cast<uint64_t>(work_rest));               \
                    convertToOutputTypePhilox(res, m_min_val.P, m_range_val.P, out_t, el_to_copy);      \
                    if (++n == 0) {                                                                     \
                        counter++;                                                                      \
                    }                                                                                   \
                }                                                                                       \
            } break;

            switch (m_output_prc) {
                EXEC_CASE(f32)
                EXEC_CASE(f16)
                EXEC_CASE(bf16)
                EXEC_CASE(i32)
                EXEC_CASE(i64)
                default: THROW_CPU_NODE_ERR("Unsupported type of RandomUniform: ", m_output_prc.to_string());
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

    return { n_state, counter_state };
}

////////////// MERSENNE algo ///////////////

namespace {

constexpr int32_t MERSENNE_STATE_N = 624;
constexpr int32_t MERSENNE_STATE_M = 397;

uint32_t twist(uint32_t u, uint32_t v) {
    return (((u & 0x80000000) | (v & 0x7fffffff)) >> 1) ^ (v & 1 ? 0x9908b0df : 0);
}

inline void next_mersenne_state(uint32_t* mersenne_state) {
    auto* current_state_ptr = mersenne_state;
    for (int j = MERSENNE_STATE_N - MERSENNE_STATE_M + 1; --j; current_state_ptr++) {
        *current_state_ptr = current_state_ptr[MERSENNE_STATE_M] ^ twist(current_state_ptr[0], current_state_ptr[1]);
    }

    for (int j = MERSENNE_STATE_M; --j; mersenne_state++) {
        *current_state_ptr = current_state_ptr[MERSENNE_STATE_M - MERSENNE_STATE_N] ^ twist(current_state_ptr[0], current_state_ptr[1]);
    }

    *current_state_ptr = current_state_ptr[MERSENNE_STATE_M - MERSENNE_STATE_N] ^ twist(current_state_ptr[0], mersenne_state[0]);
}

void runMersenneTwister(uint32_t* random_numbers, uint32_t* mersenne_state, uint64_t mersenne_state_start_idx, uint64_t work_id) {
    // Unwarped loop for optimal performance, standardized to 4 random uint32s.
    result[0] = *(mersenne_state[mersenne_state_start_idx + 4 * work_id]);
    result[0] ^= (result[0] >> 11);
    result[0] ^= (result[0] << 7) & 0x9d2c5680;
    result[0] ^= (result[0] << 15) & 0xefc60000;
    result[0] ^= (result[0] >> 18);

    result[1] = *(mersenne_state[mersenne_state_start_idx + 4 * work_id + 1]);
    result[1] ^= (result[1] >> 11);
    result[1] ^= (result[1] << 7) & 0x9d2c5680;
    result[1] ^= (result[1] << 15) & 0xefc60000;
    result[1] ^= (result[1] >> 18);

    result[2] = *(mersenne_state[mersenne_state_start_idx + 4 * work_id + 2]);
    result[2] ^= (result[2] >> 11);
    result[2] ^= (result[2] << 7) & 0x9d2c5680;
    result[2] ^= (result[2] << 15) & 0xefc60000;
    result[2] ^= (result[2] >> 18);

    result[3] = *(mersenne_state[mersenne_state_start_idx + 4 * work_id + 3]);
    result[3] ^= (result[3] >> 11);
    result[3] ^= (result[3] << 7) & 0x9d2c5680;
    result[3] ^= (result[3] << 15) & 0xefc60000;
    result[3] ^= (result[3] >> 18);
}

inline void convertToOutputTypeMersenne(const uint32_t* in,
                                float min,
                                float range,
                                float* out,
                                size_t elements_remaining) {
    const auto mask = static_cast<uint32_t>((uint64_t(1) << std::numeric_limits<float>::digits) - 1);
    const auto divisor = static_cast<float>(1) / (uint64_t(1) << std::numeric_limits<float>::digits);

    for (size_t i = 0lu; i < std::min(4, elements_remaining); i++) {
        const float ret = (in[i] & mask) * divisor;
        out[i] = ret * range + min;
    }
}

inline void convertToOutputTypeMersenne(const uint32_t* in,
                                float16 min,
                                float16 range,
                                float16* out,
                                size_t elements_remaining) {
    const auto mask = static_cast<uint32_t>((uint64_t(1) << std::numeric_limits<float16>::digits) - 1);
    const auto divisor = static_cast<float>(1) / (uint64_t(1) << std::numeric_limits<float16>::digits);

    for (size_t i = 0lu; i < std::min(4, elements_remaining); i++) {
        const float ret = (in[i] & mask) * divisor;
        out[i] = ret * range + min;
    }
}

inline void convertToOutputTypeMersenne(const uint32_t* in,
                                bfloat16 min,
                                bfloat16 range,
                                bfloat16* out,
                                size_t elements_remaining) {
    const auto mask = static_cast<uint32_t>((1UL << 8) - 1);
    const auto divisor = static_cast<float>(1) / (1UL << 8);

    for (size_t i = 0lu; i < std::min(4, elements_remaining); i++) {
        const float ret = (in[i] & mask) * divisor;
        out[i] = ret * range + min;
    }
}

inline void convertToOutputTypeMersenne(const uint32_t* in,
                                int32_t min,
                                int32_t range,
                                int32_t* out,
                                size_t elements_remaining) {
    for (size_t i = 0lu; i < std::min(4, elements_remaining); i++) {
        out[i] = static_cast<int32_t>(in[i] % range + min);
    }
}

inline void convertToOutputTypeMersenne(const uint32_t* in,
                                int64_t min,
                                int64_t range,
                                int64_t* out,
                                size_t elements_remaining) {
    if (m_mersenne_twister_optimization_enabled) {
        for (size_t i = 0lu; i < std::min(4, elements_remaining); i++) {
            out[i] = static_cast<int32_t>(in[i] % range + min);
        }
    } else {
        for (size_t i = 0lu; i < std::min(2, elements_remaining); i++) {
            out[i] = static_cast<int64_t>(((static_cast<uint64_t>(in[i * 2]) << 32) + in[i * 2 + 1]) % range + min);
        }       
    }

}
} // namespace

void RandomUniform::computeMersenneTwister(void* out, size_t out_el_num) {
    // When both seed values are equal to zero RandomUniform should generate non-deterministic sequence.
    if (m_global_seed == 0lu && m_op_seed == 0lu) {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        m_global_seed = std::rand();
    }

    auto out_u8 = reinterpret_cast<uint8_t*>(out);
    uint32_t mersenne_state[MERSENNE_STATE_N];

    const auto state_regenerations_required = static_cast<uint64_t>(std::ceil(static_cast<float>(out_el_num) / static_cast<float>(MERSENNE_STATE_N / m_elements_generated)));

    if (m_jit_kernel) {
#if defined(OPENVINO_ARCH_X86_64)
        for (uint64_t state_id = 0; state_id < state_regenerations_required; ++state_id) {
            next_mersenne_state(mersenne_state);
            parallel_nt(m_threads_num, [&](const int ithr, const int nthr) {
                auto& params = m_thread_params[ithr];
                if (params.work_amount == 0lu) {
                    return;
                }
                auto el_remin = out_el_num - MERSENNE_STATE_N / m_elements_generated * state_nr - ithr * m_elements_generated;

                kernel::random_uniform::MersenneTwisterGeneratorCallArgs args;

                args.dst_ptr     = (out_u8 + params.dst_shift);
                args.state_ptr    = &mersenne_state;
                args.min_ptr     = &m_min_val;
                args.range_ptr   = &m_range_val;
                args.state_id    = state_id;
                args.state_shift = params.state_shift;
                args.step        = params.step;
                args.work_amount = params.work_amount;
                args.elements_remaining = el_remin;
                args.optimization_enabled = m_optimization_enabled;

                (*m_jit_kernel)(&args);
            });
        }
#endif // OPENVINO_ARCH_X86_64
    } else {
        for (uint64_t state_nr = 0; state_id < state_regenerations_required; ++state_nr) {
            next_mersenne_state(mersenne_state);
            auto thread_body = [&](const int ithr, const int nthr) {
                auto& params = m_thread_params[ithr];
                
                auto out_state_shift = state_nr * MERSENNE_STATE_N * m_output_prc.size();
                auto out_cur = out_u8 + out_state_shift + params.dst_shift;
                auto work_amount = static_cast<int64_t>(params.work_amount);
                auto mersenne_state_start_idx = params.state_shift;
                auto step = params.step;
                uint32_t random_numbers[4];
                auto el_remain = out_el_num - MERSENNE_STATE_N / m_elements_generated * state_nr - ithr * m_elements_generated;

                if (work_amount == 0lu) {
                    return;
                }

#define EXEC_CASE(P)                                                                                                    \
                case element::P: {                                                                                      \
                    auto output_ptr = reinterpret_cast<element_type_traits<element::P>::value_type *>(out_cur);         \
                    for (auto work_id = 0; work_id < work_amount; work_id++, output_ptr += step) {                      \
                        runMersenneTwister(random_numbers, mersenne_state, mersenne_state_start_idx, work_id);          \
                        convertToOutputTypeMersenne(random_numbers, m_min_val.P, m_range_val.P, output_ptr, el_remain); \
                    }                                                                                                   \
                } break;

                switch (m_output_prc) {
                    EXEC_CASE(f32)
                    EXEC_CASE(f16)
                    EXEC_CASE(bf16)
                    EXEC_CASE(i32)
                    EXEC_CASE(i64)
                    default: THROW_CPU_NODE_ERR("Unsupported type of RandomUniform: ", m_output_prc.to_string());
                }
#undef EXEC_CASE
            };
            parallel_nt(m_threads_num, thread_body);
        }
    }
}

////////////// STL algo ///////////////

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
                    std::uniform_real_distribution<float>{m_min_val.f32, m_max_val.f32}, out, work_amount);
        } break;
        case element::i32: {
            generateData<int32_t, std::uniform_int_distribution<int32_t>>(
                    std::uniform_int_distribution<int32_t>{m_min_val.i32, m_max_val.i32}, out, work_amount);
        } break;
        case element::i64: {
            generateData<int64_t, std::uniform_int_distribution<int64_t>>(
                    std::uniform_int_distribution<int64_t>{m_min_val.i64, m_max_val.i64}, out, work_amount);
        } break;
        default:
            THROW_CPU_NODE_ERR("has unsupported output type: ", m_output_prc);
    }
}

//////////////////////////////////

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
