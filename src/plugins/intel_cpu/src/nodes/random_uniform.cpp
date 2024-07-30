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
        if (as_type_ptr<const op::v8::RandomUniform>(op)->get_alignment() != op::PhiloxAlignment::TENSORFLOW) {
            errorMessage = "Only TENSORFLOW alignment mode is supported by the CPU plugin.";
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
    if (out_prc.is_real() && ((m_algo == PHILOX &&
            !one_of(out_prc, ov::element::f32, ov::element::f16, ov::element::bf16)) ||
            (m_algo == STL && !one_of(out_prc, ov::element::f32)))) {
        out_prc = ov::element::f32;
    }
    if (!out_prc.is_real() && !one_of(out_prc, ov::element::i32, ov::element::i64)) {
        out_prc = ov::element::i32;
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

    if (m_algo == PHILOX) {
#if defined(OPENVINO_ARCH_X86_64)
        kernel::RandomUniformCompileParams jcp;

        jcp.out_data_type = m_output_prc;

        m_jit_kernel = kernel::JitKernel<kernel::RandomUniformCompileParams, kernel::RandomUniformCallArgs>::createInstance<kernel::RandomUniform>(jcp);

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

        if (m_out_el_num < PHILOX_PARALLEL_EXECUTION_THRESHOLD) {
            m_threads_num = 1;
        } else {
            m_threads_num = parallel_get_max_threads();
        }
        m_thread_params.resize(m_threads_num);

        parallel_nt(m_threads_num, [&](const int ithr, const int nthr) {
            auto& p = m_thread_params[ithr];
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

                p.step = m_output_prc.size() > 4 ? 2 : 4;
            }

            if (end > m_out_el_num) {
                end = m_out_el_num;
            }
            if (start > end) {
                start = end;
            }
            p.work_amount = end - start;
            p.n_shift = start / PHILOX_GROUP_SIZE;
            p.dst_shift = start * m_output_prc.size();
        });
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
    } else if (m_algo == STL) {
        computeStl(data, m_out_el_num);
    } else {
        THROW_CPU_NODE_ERR("unsupported algorithm.");
    }
}

void RandomUniform::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
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

    for (size_t i = 0lu; i < ROUNDS_NUMBER; i++) {
        calculateRound(key_32, counter_32, n_32);
        if (i < ROUNDS_NUMBER - 1)
            raiseKey(key_32);
    }

    res[0] = n_32[0];
    res[1] = n_32[1];
    res[2] = counter_32[0];
    res[3] = counter_32[1];
}

inline void convertToOutputType(const uint32_t* in,
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

inline void convertToOutputType(const uint32_t* in,
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

inline void convertToOutputType(const uint32_t* in,
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

inline void convertToOutputType(const uint32_t* in,
                                int32_t min,
                                int32_t range,
                                int32_t* out,
                                size_t el_to_copy) {
    for (size_t i = 0lu; i < el_to_copy; i++) {
        out[i] = static_cast<int32_t>(in[i] % range + min);
    }
}

inline void convertToOutputType(const uint32_t* in,
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
                auto& p = m_thread_params[ithr];
                if (p.work_amount == 0lu) {
                    return;
                }
                auto n = n_state + p.n_shift;

                kernel::RandomUniformCallArgs args;

                args.dst_ptr     = (out_u8 + p.dst_shift);
                args.key_ptr     = &m_global_seed;
                args.counter_ptr = &counter;
                args.n_ptr       = &n;
                args.min_ptr     = &m_min_val;
                args.range_ptr   = &m_range_val;
                args.work_amount = p.work_amount;

                (*m_jit_kernel)(&args);
            });
#endif // OPENVINO_ARCH_X86_64
    } else {
        auto threadBody = [&](const int ithr, const int nthr) {
            auto& p = m_thread_params[ithr];
            if (p.work_amount == 0lu) {
                return;
            }
            auto n = n_state + p.n_shift;
            auto out_cur = out_u8 + p.dst_shift;
            auto work_rest = static_cast<int64_t>(p.work_amount);
            uint32_t res[4];

#define EXEC_CASE(P)                                                                                    \
            case element::P: {                                                                          \
                auto out_t = reinterpret_cast<element_type_traits<element::P>::value_type *>(out_cur);  \
                for (; work_rest > 0l; work_rest -= p.step, out_t += p.step) {                          \
                    runPhilox(m_global_seed, counter, n, res);                                          \
                    auto el_to_copy = std::min(p.step, static_cast<uint64_t>(work_rest));               \
                    convertToOutputType(res, m_min_val.P, m_range_val.P, out_t, el_to_copy);            \
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

////////////// STL algo ///////////////
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

template <typename T, typename DISTR_TYPE>
void RandomUniform::generateData(DISTR_TYPE distribution, void* out, size_t work_amount) {
    auto dst = reinterpret_cast<T*>(out);
    for (size_t i = 0; i < work_amount; i++) {
        *dst = distribution(m_generator);
        dst++;
    }
}
//////////////////////////////////

void RandomUniform::initEdgeValues(OutputType& dst, const void* src, const element::Type& output_type) {
#define EL_CASE(E) \
    case element::E: \
        dst.E = *reinterpret_cast<const element_type_traits<element::E>::value_type *>(src); \
        break;

    switch (output_type) {
        EL_CASE(f32)
        EL_CASE(f16)
        EL_CASE(bf16)
        EL_CASE(i32)
        EL_CASE(i64)
        EL_CASE(f64)
        default:
            THROW_CPU_NODE_ERR("has unsupported output precision: ", output_type);
    }

#undef EL_CASE
}

void RandomUniform::evalRange() {
#define EL_CASE(E) \
    case element::E: \
        m_range_val.E = m_max_val.E - m_min_val.E; \
        break;

    switch (m_output_prc) {
        EL_CASE(f32)
        EL_CASE(f16)
        EL_CASE(bf16)
        EL_CASE(i32)
        EL_CASE(i64)
        EL_CASE(f64)
        default:
            THROW_CPU_NODE_ERR("has unsupported output precision: ", m_output_prc);
    }

#undef EL_CASE
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

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
