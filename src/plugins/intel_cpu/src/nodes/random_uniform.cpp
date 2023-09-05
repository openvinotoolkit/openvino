// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "random_uniform.hpp"

#include "ie_parallel.hpp"
#include "ie_ngraph_utils.hpp"
#include <openvino/op/constant.hpp>
#include <openvino/op/random_uniform.hpp>
#include "shape_inference/custom/random_uniform.hpp"

using namespace dnnl::impl::cpu;

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
        : Node(op, context, RandomUniformShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        THROW_CPU_NODE_ERR(errorMessage);
    }

    // RandomUniform should generate new sequence each run even if all inputs are constants. So that method Node::IsConstant()
    // doesn't return 'True' for RandomUniform with all constant inputs and the node generates new values for each inference,
    // we set 'NoConst' value for 'ConstantType' in ctor.
    constant = ConstantType::NoConst;

    auto rnd_op = as_type_ptr<op::v8::RandomUniform>(op);
    m_global_seed = rnd_op->get_global_seed();
    m_op_seed = rnd_op->get_op_seed();

    m_output_prc = op->get_output_element_type(0);

    for (size_t i = 0lu; i < op->get_input_size(); i++) {
        if (is_type<op::v0::Constant>(op->get_input_node_ptr(i))) {
            m_const_inputs[i] = true;
        }
    }

    if (m_const_inputs[MIN_VAL]) {
        initEdgeValues(m_min_val, as_type<op::v0::Constant>(op->get_input_node_ptr(MIN_VAL))->get_data_ptr(), m_output_prc);
    }
    if (m_const_inputs[MAX_VAL]) {
        initEdgeValues(m_max_val, as_type<op::v0::Constant>(op->get_input_node_ptr(MAX_VAL))->get_data_ptr(), m_output_prc);
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
    if (!one_of(shape_prc, InferenceEngine::Precision::I32, InferenceEngine::Precision::I64)) {
        shape_prc = InferenceEngine::Precision::I32;
    }

    auto out_prc = getOriginalOutputPrecisionAtPort(0);
    if (out_prc.is_float() && !one_of(out_prc, InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16, InferenceEngine::Precision::BF16)) {
        out_prc = InferenceEngine::Precision::FP32;
    }
    if (!out_prc.is_float() && !one_of(out_prc, InferenceEngine::Precision::I32, InferenceEngine::Precision::I64)) {
        out_prc = InferenceEngine::Precision::I32;
    }
    m_output_prc = InferenceEngine::details::convertPrecision(out_prc);

    addSupportedPrimDesc({{LayoutType::ncsp, shape_prc, m_const_inputs[SHAPE]},
                          {LayoutType::ncsp, out_prc, m_const_inputs[MIN_VAL]},
                          {LayoutType::ncsp, out_prc, m_const_inputs[MAX_VAL]}},
                         {{LayoutType::ncsp, out_prc}},
                         ref_any);
}

void RandomUniform::createPrimitive() {
    if (m_algo == PHILOX) {
#if defined(OPENVINO_ARCH_X86_64)
        kernel::RandomUniformCompileParams jcp;

        jcp.out_data_type = m_output_prc;

        m_jit_kernel = kernel::JitKernel<kernel::RandomUniformCompileParams, kernel::RandomUniformCallArgs>::createInstance<kernel::RandomUniform>(jcp);

        if (m_jit_kernel) {
            if (auto selected_pd = getSelectedPrimitiveDescriptor()) {
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
    if (m_out_shape != getChildEdgeAt(0)->getMemoryPtr()->getShape().getStaticDims()) {
        return true;
    }
    return false;
}

void RandomUniform::prepareParams() {
    m_out_shape = getChildEdgeAt(0)->getMemoryPtr()->getShape().getStaticDims();
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
                const auto block_size = (m_jit_kernel->getVectorLen() / m_output_prc.size()) * 2;
                const auto blocks_num = (m_out_el_num + block_size - 1) / block_size;
                const auto blocks_per_thr = (blocks_num + nthr - 1) / nthr;

                start = ithr * blocks_per_thr * block_size;
                end = (ithr + 1) * blocks_per_thr * block_size;
            } else {
                const auto groups_num = (m_out_el_num + PHILOX_GROUP_SIZE - 1) / PHILOX_GROUP_SIZE;
                const auto groups_per_thr = (groups_num + nthr - 1) / nthr;

                start = ithr * groups_per_thr * PHILOX_GROUP_SIZE;
                end = (ithr + 1) * groups_per_thr * PHILOX_GROUP_SIZE;

                p.step = m_output_prc.size() > 4 ? 2 : 4;
                p.step_b = p.step * m_output_prc.size();
            }

            if (end > m_out_el_num) {
                end = m_out_el_num;
            }
            if (start > end) {
                start = end;
            }
            p.work_amount = end - start;
            p.work_amount_b = static_cast<int64_t>(p.work_amount * m_output_prc.size());
            p.n_shift = start / PHILOX_GROUP_SIZE;
            p.dst_shift = start * m_output_prc.size();
        });
    }
}

void RandomUniform::execute(dnnl::stream strm) {
    if (!m_const_inputs[MIN_VAL]) {
        initEdgeValues(m_min_val, getParentEdgeAt(MIN_VAL)->getMemoryPtr()->getData(), m_output_prc);
    }
    if (!m_const_inputs[MAX_VAL]) {
        initEdgeValues(m_max_val, getParentEdgeAt(MAX_VAL)->getMemoryPtr()->getData(), m_output_prc);
    }

    auto data = getChildEdgeAt(0)->getMemoryPtr()->getData();

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

uint64_t uniteHighLow(uint32_t high, uint32_t low) {
    return (static_cast<uint64_t>(high) << 32) + low;
}

void calculateRound(const uint32_t* key, uint32_t* counter, uint32_t* n) {
    uint64_t prod_0 = STATISTIC_MAXIMIZING_MULTIPLIER_N * n[0];
    uint64_t prod_1 = STATISTIC_MAXIMIZING_MULTIPLIER_COUNTER * counter[0];
    n[0] = static_cast<uint32_t>(prod_1 >> 32) ^ n[1] ^ key[0];
    n[1] = static_cast<uint32_t>(prod_1);
    counter[0] = static_cast<uint32_t>(prod_0 >> 32) ^ counter[1] ^ key[1];
    counter[1] = static_cast<uint32_t>(prod_0);
}

void raiseKey(uint32_t* key) {
    key[0] += CRUSH_RESISTANCE_CONST_LOWER_VALUE;
    key[1] += CRUSH_RESISTANCE_CONST_UPPER_VALUE;
}

float uint32ToFloat(uint32_t x) {
    RandomUniform::OutputType out_val;
    out_val.u32 = (static_cast<uint32_t>(127) << 23) | (x & 0x7fffffu);
    return out_val.f32 - 1.0f;
}

float16 uint32ToFloat16(uint32_t x) {
    RandomUniform::OutputType out_val;
    uint16_t x_uint16 = static_cast<uint16_t>(x);
    out_val.u16 = (static_cast<uint16_t>(15) << 10) | (x_uint16 & 0x3ffu);
    return out_val.f16 - static_cast<float16>(1);
}

bfloat16 uint32ToBfloat16(uint32_t x) {
    RandomUniform::OutputType out_val;
    uint16_t x_uint16 = static_cast<uint16_t>(x);
    out_val.u16 = (static_cast<uint16_t>(127) << 7) | (x_uint16 & 0x7fu);
    return out_val.bf16 - static_cast<bfloat16>(1);
}

void runPhilox(uint64_t key, uint64_t counter, uint64_t n, uint32_t* res) {
    uint32_t* key_32 = reinterpret_cast<uint32_t*>(&key);
    uint32_t* counter_32 = reinterpret_cast<uint32_t*>(&counter);
    uint32_t* n_32 = reinterpret_cast<uint32_t*>(&n);

    for (size_t i = 0; i < ROUNDS_NUMBER; i++) {
        calculateRound(key_32, counter_32, n_32);
        if (i < ROUNDS_NUMBER - 1)
            raiseKey(key_32);
    }

    res[0] = n_32[0];
    res[1] = n_32[1];
    res[2] = counter_32[0];
    res[3] = counter_32[1];
}

template <typename T>
void convertToOutputType(const uint32_t* res,
                         size_t step,
                         const element::Type& elem_type,
                         T min_val,
                         T max_val,
                         uint8_t* out,
                         size_t num_to_copy,
                         T (*convert_single_input)(uint32_t) = nullptr,
                         T (*convert_two_inputs)(uint32_t, uint32_t, T, T) = nullptr,
                         T (*mod_func)(uint32_t, T, T) = nullptr) {
    std::vector<T> res_out_type(step);

    if (elem_type.size() > 4) {
        res_out_type[0] = convert_two_inputs(res[0], res[1], min_val, max_val);
        res_out_type[1] = convert_two_inputs(res[2], res[3], min_val, max_val);
    } else {
        std::transform(res,
                       res + step,
                       res_out_type.data(),
                       [&min_val, &max_val, &convert_single_input, &mod_func](uint32_t elem) {
                           if (convert_single_input != nullptr) {
                               return convert_single_input(elem) * (max_val - min_val) + min_val;
                           } else {
                               return mod_func(elem, min_val, max_val);
                           }
                       });
    }

    memcpy(out, res_out_type.data(), num_to_copy);
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
                args.max_ptr     = &m_max_val;
                args.work_amount = p.work_amount;

                (*m_jit_kernel)(&args);
            });
    } else {
        auto threadBody = [&](const int ithr, const int nthr) {
            auto& p = m_thread_params[ithr];
            if (p.work_amount == 0lu) {
                return;
            }
            auto n = n_state + p.n_shift;
            auto out_cur = out_u8 + p.dst_shift;
            auto work_rest = p.work_amount_b;
            uint32_t res[4];

            for (; work_rest > 0; work_rest -= p.step_b, out_cur += p.step_b) {
                runPhilox(m_global_seed, counter, n, res);
                auto bytes_to_copy = std::min(p.step_b, static_cast<uint64_t>(work_rest));

                switch (m_output_prc) {
                    case element::f32: {
                        convertToOutputType<float>(res, p.step, m_output_prc, m_min_val.f32, m_max_val.f32, out_cur, bytes_to_copy, uint32ToFloat);
                    } break;
                    case element::f16: {
                        convertToOutputType<float16>(res, p.step, m_output_prc, m_min_val.f16, m_max_val.f16, out_cur, bytes_to_copy, uint32ToFloat16);
                    } break;
                    case element::bf16: {
                        convertToOutputType<bfloat16>(res, p.step, m_output_prc, m_min_val.bf16, m_max_val.bf16, out_cur, bytes_to_copy, uint32ToBfloat16);
                    } break;
                    case element::i32: {
                        convertToOutputType<int>(res, p.step, m_output_prc, m_min_val.i32, m_max_val.i32, out_cur, bytes_to_copy, nullptr, nullptr,
                                                [](uint32_t x, int mn, int mx) {
                                                    return static_cast<int>(x % (mx - mn) + mn);
                                                });
                    } break;
                    case element::i64: {
                        convertToOutputType<int64_t>(res, p.step, m_output_prc, m_min_val.i64, m_max_val.i64, out_cur, bytes_to_copy, nullptr,
                                                [](uint32_t a, uint32_t b, int64_t mn, int64_t mx) {
                                                    return static_cast<int64_t>(uniteHighLow(b, a) % (mx - mn) + mn);
                                                });
                    } break;
                    default: THROW_CPU_NODE_ERR("Unsupported type of RandomUniform: ", m_output_prc.to_string());
                }

                if (++n == 0) {
                    counter++;
                }
            }
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
        case element::f64: {
            generateData<double, std::uniform_real_distribution<double>>(
                    std::uniform_real_distribution<double>{m_min_val.f64, m_max_val.f64}, out, work_amount);
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
        if (selectedPrimitiveDesc->getConfig().outConfs[0].getMemDesc()->getPrecision() != InferenceEngine::Precision::U8) {
            str_type += "_" + std::string(selectedPrimitiveDesc->getConfig().outConfs[0].getMemDesc()->getPrecision().name());
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
