// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "embedding_bag_sum.hpp"
#include "ie_parallel.hpp"
#include "jit_generator.hpp"
#include "list.hpp"

#include <set>
#include <string>
#include <vector>

using namespace InferenceEngine;
using namespace InferenceEngine::Extensions::Cpu;
using namespace mkldnn::impl;
using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::utils;


#define GET_OFF(field) offsetof(jit_emb_bag_call_args, field)

template <cpu_isa_t isa>
struct jit_uni_embedding_bag_sum_kernel_f32 : public jit_uni_embedding_bag_sum_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_embedding_bag_sum_kernel_f32)

    explicit jit_uni_embedding_bag_sum_kernel_f32(jit_emb_bag_config_params jcp) : jit_uni_embedding_bag_sum_kernel(jcp), jit_generator() {
        this->preamble();

        mov(reg_src, ptr[reg_params + GET_OFF(src)]);
        mov(reg_dst, ptr[reg_params + GET_OFF(dst)]);;
        mov(reg_ww, ptr[reg_params + GET_OFF(with_weights)]);

        if (_jcp.with_weights) {
            Xbyak::Label skip_weights;
            cmp(reg_ww, 0);
            je(skip_weights, T_NEAR);
            mov(reg_weights, ptr[reg_params + GET_OFF(weights)]);
            if (isa == cpu::avx512_common) {
                vbroadcastss(v_weights, ptr[reg_weights]);
            } else {
                uni_vbroadcastss(v_weights, ptr[reg_weights]);
            }
            L(skip_weights);
        }

        size_t v_len = 16;
        if (isa == cpu::avx512_common) {
            const size_t stride = v_len * sizeof(float);
            for (size_t i = v_len; i <= _jcp.emb_dim; i += v_len) {
                vmovups(v_dst, zword[reg_dst]);
                vmovups(v_src, zword[reg_src]);
                if (_jcp.with_weights) {
                    Xbyak::Label to_add, to_end;
                    cmp(reg_ww, 0);
                    je(to_add, T_NEAR);
                        vfmadd231ps(v_dst, v_src, v_weights);
                    jmp(to_end);
                    L(to_add);
                        vaddps(v_dst, v_dst, v_src);
                    L(to_end);
                } else {
                    vaddps(v_dst, v_dst, v_src);
                }
                vmovups(zword[reg_dst], v_dst);
                add(reg_src, stride);
                add(reg_dst, stride);
            }
        } else {
            v_len = 8;
            if (isa == sse42)
                v_len = 4;
            const size_t stride = v_len * sizeof(float);;
            for (size_t i = v_len; i <= _jcp.emb_dim; i += v_len) {
                uni_vmovups(v_dst, ptr[reg_dst]);
                uni_vmovups(v_src, ptr[reg_src]);
                if (_jcp.with_weights) {
                    Xbyak::Label to_add, to_end;
                    cmp(reg_ww, 0);
                    je(to_add, T_NEAR);
                        uni_vfmadd231ps(v_dst, v_src, v_weights);
                        jmp(to_end);
                    L(to_add);
                        uni_vaddps(v_dst, v_dst, v_src);
                    L(to_end);
                } else {
                    uni_vaddps(v_dst, v_dst, v_src);
                }
                uni_vmovups(ptr[reg_dst], v_dst);
                add(reg_src, stride);
                add(reg_dst, stride);
            }
        }

        /*size_t work_rest = _jcp.emb_dim % v_len;
        if (work_rest > 0) {
            const size_t stride = sizeof(float);
            if (_jcp.with_weights) {
                fld(ptr[reg_weights]);
            }
            for (size_t i = 0; i < work_rest; i++) {
                //mov(reg_v, ptr[reg_src]);
                fld(ptr[reg_src]);
                if (_jcp.with_weights) {
                    fmul(st0, st1);
                }
                fadd(ptr[reg_dst]);
                fstp(ptr[reg_dst]);
                add(reg_src, stride);
                add(reg_dst, stride);
            }
        }*/

        this->postamble();
        _ker = (decltype(_ker)) this->getCode();
    }

protected:
    using Vmm = typename conditional3<isa == avx512_common, Xbyak::Zmm, isa == avx2, Xbyak::Ymm, Xbyak::Xmm>::type;

    Vmm v_weights = Vmm(13);
    Vmm v_src = Vmm(14);
    Vmm v_dst = Vmm(15);

    Xbyak::Reg64 reg_v = r11;
    Xbyak::Reg64 reg_w = r12;
    Xbyak::Reg64 reg_ww = r13;
};


const std::set<size_t> MKLDNNEmbeddingBagSum::_supported_indexes_type_size = {sizeof(INT32), sizeof(INT64)};

MKLDNNEmbeddingBagSum::MKLDNNEmbeddingBagSum(
        const CNNLayer* layer,
        size_t required_input_num,
        size_t indices_idx,
        size_t per_sample_weights_idx,
        size_t default_index_idx,
        const std::set<Precision>& supported_precisions) :
            INDICES_IDX(indices_idx),
            PER_SAMPLE_WEIGHTS_IDX(per_sample_weights_idx),
            DEFAULT_INDEX_IDX(default_index_idx) {
    try {
        std::string log_prefix = "Layer EmbeddingBagSum with name '";
        if (layer->insData.size() < required_input_num || layer->outData.size() != 1)
            THROW_IE_EXCEPTION << log_prefix << layer->name << "' has incorrect number of input or output edges!";
        _l_name = layer->name;

        auto inData = layer->insData[0].lock();
        auto indicesData = layer->insData[INDICES_IDX].lock();
        if (inData == nullptr || indicesData == nullptr)
            THROW_IE_EXCEPTION << log_prefix << _l_name << "' layer has nullable input data.";

        const auto precision = inData->getTensorDesc().getPrecision();
        if (!supported_precisions.empty()) {
            if (supported_precisions.find(precision) == supported_precisions.end())
                THROW_IE_EXCEPTION << log_prefix << _l_name << "' layer has unsupported precision: " << precision.name();
        } else {
            static const std::set<Precision> default_supported_precisions =
                {Precision::FP32, Precision::BF16, Precision::I8, Precision::U8, Precision::I32};
            if (default_supported_precisions.find(precision) == default_supported_precisions.end())
                THROW_IE_EXCEPTION << log_prefix << _l_name << "' layer has unsupported precision: " << precision.name();
        }

        if (layer->insData.size() > PER_SAMPLE_WEIGHTS_IDX)
            _with_weights = true;
        if (_with_weights) {
            auto weightsData = layer->insData[PER_SAMPLE_WEIGHTS_IDX].lock();
            if (weightsData == nullptr)
                 THROW_IE_EXCEPTION << log_prefix << _l_name << "' layer has nullable weights data";
            if (weightsData->getTensorDesc().getDims() != indicesData->getTensorDesc().getDims())
                 THROW_IE_EXCEPTION << log_prefix << _l_name << "' layer must have equal shapes for indices and per_sample_weights inputs.";
        }

        LayerConfig config;
        config.inConfs.resize(layer->insData.size());
        for (int i = 0; i < layer->insData.size(); i++) {
            auto data = layer->insData[i].lock();
            if (data == nullptr)
                THROW_IE_EXCEPTION << log_prefix << _l_name << "' layer has nullable input data";
            config.inConfs[i].desc = TensorDesc(data->getTensorDesc());
        }

        DataConfig outConfig;
        outConfig.desc = TensorDesc(inData->getTensorDesc().getPrecision(),
            layer->outData[0]->getTensorDesc().getDims(),
            layer->outData[0]->getTensorDesc().getLayout());
        config.outConfs.push_back(outConfig);
        config.dynBatchSupport = false;

        confs.push_back(config);

        const auto& inDataDims = inData->getTensorDesc().getDims();
        const size_t IN_DATA_DIMS_SIZE = inDataDims.size();
        _multipliers.resize(IN_DATA_DIMS_SIZE);
        _multipliers[IN_DATA_DIMS_SIZE - 1] = 1lu;
        for (int i = IN_DATA_DIMS_SIZE - 1; i > 0; i--) {
            _multipliers[i - 1] = inDataDims[i] * _multipliers[i];
        }

        if (inDataDims.size() == 2 && inData->getTensorDesc().getPrecision() == Precision::FP32) {
            auto jcp = jit_emb_bag_config_params();
            jcp.emb_dim = inDataDims[1];
            jcp.with_weights = _with_weights;

            if (mayiuse(cpu::avx512_common)) {
                emb_bag_kernel.reset(new jit_uni_embedding_bag_sum_kernel_f32<cpu::avx512_common>(jcp));
            } else if (mayiuse(cpu::avx2)) {
                emb_bag_kernel.reset(new jit_uni_embedding_bag_sum_kernel_f32<cpu::avx2>(jcp));
            } else if (mayiuse(cpu::sse42)) {
                emb_bag_kernel.reset(new jit_uni_embedding_bag_sum_kernel_f32<cpu::sse42>(jcp));
            }
        }
    } catch (InferenceEngine::details::InferenceEngineException &ex) {
        errorMsg = ex.what();
    }
}

StatusCode MKLDNNEmbeddingBagSum::execute(
        std::vector<Blob::Ptr>& inputs,
        std::vector<Blob::Ptr>& outputs,
        ResponseDesc *resp) noexcept {
    switch (inputs[0]->getTensorDesc().getPrecision()) {
        case Precision::FP32:
        case Precision::BF16: {
            process_data<PrecisionTrait<Precision::FP32>::value_type>(inputs, outputs);
            break;
        }
        case Precision::I8: {
            process_data<PrecisionTrait<Precision::I8>::value_type>(inputs, outputs);
            break;
        }
        case Precision::U8: {
            process_data<PrecisionTrait<Precision::U8>::value_type>(inputs, outputs);
            break;
        }
        case Precision::I32: {
            process_data<PrecisionTrait<Precision::I32>::value_type>(inputs, outputs);
            break;
        }
        default: {
            if (resp) {
                std::string errorMsg = "EmbeddingBagSum layer does not support precision '"
                        + std::string(inputs[0]->getTensorDesc().getPrecision().name()) + "'";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return GENERAL_ERROR;
        }
    }

    return OK;
}

template<typename T>
void MKLDNNEmbeddingBagSum::process_data(
        std::vector<Blob::Ptr>& inputs,
        std::vector<Blob::Ptr>& outputs) noexcept {
    const T* src_data = inputs[0]->cbuffer().as<const T*>() +
        inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
    T* dst_data = outputs[0]->buffer().as<T*>() +
        outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
    const T* weights_data = nullptr;
    if (_with_weights)
        weights_data = inputs[PER_SAMPLE_WEIGHTS_IDX]->cbuffer().as<const T*>();
    init_from_inputs(inputs);

    const auto& inDataDims = inputs[0]->getTensorDesc().getDims();
    const size_t IN_DATA_DIMS_SIZE = inDataDims.size();

    const size_t OUTPUT_BAGS_NUM = outputs[0]->getTensorDesc().getDims()[0];

    memset(dst_data, 0, outputs[0]->byteSize());

    std::function<void(size_t, size_t, size_t, size_t, bool)> emb_cycle =
        [&](size_t src_index, size_t dst_index, size_t emb_idx, size_t weights_idx, bool with_weights) {
            for (size_t i = 0lu; i < inDataDims[emb_idx]; i++) {
                size_t new_src_idx = src_index + i * _multipliers[emb_idx];
                size_t new_dst_idx = dst_index + i * _multipliers[emb_idx];
                if (emb_idx == IN_DATA_DIMS_SIZE - 1) {
                    if (with_weights)
                        dst_data[new_dst_idx] += src_data[new_src_idx] * weights_data[weights_idx];
                    else
                        dst_data[new_dst_idx] += src_data[new_src_idx];
                } else {
                    emb_cycle(new_src_idx, new_dst_idx, emb_idx + 1, weights_idx, with_weights);
                }
            }
        };

    auto thread_body = [&](const int ithr, const int nthr) {
        size_t start(0lu), end(0lu);
        splitter(OUTPUT_BAGS_NUM, nthr, ithr, start, end);
        if (start >= end)
            return;

        size_t indices_size = 0lu;
        const size_t* indices = nullptr;
        size_t weights_idx = 0lu;
        bool with_weights = _with_weights;

        for (size_t obi = start; obi < end; obi++) {
            size_t dst_index = obi * _multipliers[0];
            get_indices(obi, indices, indices_size, weights_idx, with_weights);
            if (indices != nullptr) {
                with_weights = with_weights & _with_weights;
                for (size_t in_idx = 0lu; in_idx < indices_size; in_idx++) {
                    if (indices[in_idx] >= inDataDims[0])
                        THROW_IE_EXCEPTION << "EmbeddingBagSum layer '" << _l_name
                            << "' has invalid embedding bag index: " << indices[in_idx];
                    size_t src_index = src_index = indices[in_idx] * _multipliers[0];
                    emb_cycle(src_index, dst_index, 1, weights_idx, with_weights);
                    weights_idx++;
                }
            }
        }
    };

    // THREAD WITH vectorization //
    auto thread_body_jit = [&](const int ithr, const int nthr) {
        size_t start(0lu), end(0lu);
        splitter(OUTPUT_BAGS_NUM, nthr, ithr, start, end);
        if (start >= end)
            return;

        size_t indices_size = 0lu;
        const size_t* indices = nullptr;
        size_t weights_idx = 0lu;
        bool with_weights = _with_weights;
        size_t v_len = 16lu;
        if (mayiuse(cpu::avx2))
            v_len = 8lu;
        else if (mayiuse(cpu::sse42))
            v_len = 4lu;
        for (size_t obi = start; obi < end; obi++) {
            size_t dst_index = obi * _multipliers[0];
            get_indices(obi, indices, indices_size, weights_idx, with_weights);
            with_weights = with_weights & _with_weights;
            if (indices != nullptr) {
                for (size_t in_idx = 0lu; in_idx < indices_size; in_idx++) {
                    if (indices[in_idx] >= inDataDims[0])
                        THROW_IE_EXCEPTION << "EmbeddingBagSum layer '" << _l_name
                            << "' has invalid embedding bag index: " << indices[in_idx];
                    size_t src_index = indices[in_idx] * _multipliers[0];
                    auto arg = jit_emb_bag_call_args();
                    arg.src = src_data + src_index;
                    arg.dst = dst_data + dst_index;
                    arg.with_weights = with_weights;
                    if (with_weights) {
                        arg.weights = weights_data + weights_idx;
                    }
                    (*emb_bag_kernel)(&arg);

                    const size_t work_rest = inDataDims[1] % v_len;
                    for (size_t i = inDataDims[1] - work_rest; i < inDataDims[1]; i++) {
                        if (with_weights)
                            dst_data[dst_index + i] += weights_data[weights_idx] * src_data[src_index + i];
                        else
                            dst_data[dst_index + i] += src_data[src_index + i];
                    }
                    if (with_weights)
                        weights_idx++;
                }
            }
        }
    };

    if (emb_bag_kernel) {
        parallel_nt(0, thread_body_jit);
    } else {
        parallel_nt(0, thread_body);
    }
}
