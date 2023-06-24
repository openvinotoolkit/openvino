// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <torch/extension.h>
#include <memory>
#include "alloca.h"
#include "module.hpp"
#include "common/utility.hpp"
#include "utility_kernel_amx.hpp"
#include "llm_emb_gpt.hpp"
#include "llm_mha_gpt.hpp"
#include "test_common.hpp"

using namespace torch::indexing;

class attn_gpt {
public:
    struct create_param {
        size_t num_heads;
        size_t head_size;
        size_t head_size_aligned;       // better to aligned to 64 bytes for best performance, apply for qkv
        size_t max_seq_len;             // max seq length for computing the size of matmul tmp result
        // supported (qkv, dst): (bf16, bf16)
        llmdnn::data_type_t qkv_precision;
        llmdnn::data_type_t dst_precision;
        size_t rotary_emb_base;
        float normal_factor;
        float rotary_pct;
        bool use_position2d;
    };
    struct exec_param {
        size_t batch;
        size_t query_seq_len;
        size_t past_seq_len;
        bool is_causal_in_attention;        // causal mask is fused in attention mask: chatglm uses it.
        uint8_t* qkv;
        uint8_t** layer_past_key_padded;
        uint8_t** layer_past_value_padded;
        int* position2d_ids;                // shape: [batch, 2, query_seq_len]
        float* attention_mask;              // attention mask, attention_mask[0] shape:
                                            //      [batch, 1, 1, key_seq_len], when is_causal_in_attention is false
                                            //      [batch, 1, query_seq_len, key_seq_len], when is_causal_in_attention is true
        uint8_t* attn_output;
        size_t head_stride_in_kv;
    };

    attn_gpt();
    bool create(const create_param& param);
    void exec(const exec_param& param);

private:
    create_param _create_param;
    std::shared_ptr<llmdnn::emb_gpt> _emb_gpt;
    std::shared_ptr<llmdnn::mha_gpt> _mha_gpt;
    std::shared_ptr<uint8_t> _query_dst;
    size_t _query_cached_batch = 0;
};

attn_gpt::attn_gpt(): _emb_gpt(std::make_shared<llmdnn::emb_gpt>()),
                      _mha_gpt(std::make_shared<llmdnn::mha_gpt>()) {

}

bool attn_gpt::create(const attn_gpt::create_param& param) {
    _create_param = param;
    llmdnn::emb_gpt::create_param emb_param;
    emb_param.num_heads = param.num_heads;
    emb_param.head_size = param.head_size;
    emb_param.head_size_aligned = param.head_size_aligned;
    emb_param.qkv_precision = param.qkv_precision;
    emb_param.dst_precision = param.dst_precision;
    emb_param.max_seq_len = param.max_seq_len;
    emb_param.rotary_emb_base = param.rotary_emb_base;
    emb_param.rotary_pct = param.rotary_pct;
    emb_param.use_position2d = param.use_position2d;

    if (!_emb_gpt->create(emb_param))
        return false;

    llmdnn::mha_gpt::create_param mha_param;
    mha_param.num_heads = param.num_heads;
    mha_param.head_size = param.head_size;
    mha_param.head_size_aligned = param.head_size_aligned;
    mha_param.normal_factor = param.normal_factor;
    mha_param.qkv_precision = param.qkv_precision;
    mha_param.dst_precision = param.dst_precision;
    mha_param.max_seq_len = param.max_seq_len;

    return _mha_gpt->create(mha_param);
}

void attn_gpt::exec(const attn_gpt::exec_param& param) {
    if (_query_cached_batch < param.batch) {
        auto capacity = param.batch * _create_param.max_seq_len * (_create_param.num_heads * _create_param.head_size_aligned) *
            llmdnn::get_precision_size(_create_param.qkv_precision);    
        _query_dst = std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(aligned_alloc(64, capacity)),
            [](void * p) { ::free(p); });
        memset(_query_dst.get(), 0, capacity);
        _query_cached_batch = param.batch;
    }

    llmdnn::emb_gpt::exec_param emb_param;
    emb_param.batch = param.batch;
    emb_param.query_seq_len = param.query_seq_len;
    emb_param.past_seq_len = param.past_seq_len;
    emb_param.qkv = param.qkv;
    emb_param.query_dst = _query_dst.get();
    emb_param.layer_past_key_padded = param.layer_past_key_padded;
    emb_param.layer_past_value_padded = param.layer_past_value_padded;
    emb_param.position2d_ids = param.position2d_ids;
    _emb_gpt->exec(emb_param);

    llmdnn::mha_gpt::exec_param mha_param;
    mha_param.batch = param.batch;
    mha_param.query_seq_len = param.query_seq_len;
    mha_param.key_seq_len = param.query_seq_len + param.past_seq_len;
    mha_param.q = emb_param.query_dst;
    mha_param.attn_output = param.attn_output;
    mha_param.head_stride_in_kv = param.head_stride_in_kv;
    mha_param.is_causal_in_attention = param.is_causal_in_attention;
    mha_param.attention_mask = param.attention_mask;
    mha_param.k = emb_param.layer_past_key_padded;
    mha_param.v = emb_param.layer_past_value_padded;
    _mha_gpt->exec(mha_param);
}

void regclass_attn_gpt(pybind11::module m) {
    py::class_<attn_gpt, std::shared_ptr<attn_gpt>> cls(m, "attn_gpt");
    cls.def(py::init<>());
    cls.def("create", [] (attn_gpt& self,
        const size_t num_heads,
        const size_t head_size,
        const size_t head_size_aligned,
        float normal_factor,
        const std::string qkv_precision_name,
        const std::string dst_precision_name,
        const size_t max_seq_len,
        const size_t rotary_emb_base,
        float rotary_pct,
        bool use_position2d) {
            attn_gpt::create_param param;
            param.num_heads = num_heads;
            param.head_size = head_size;
            param.head_size_aligned = head_size_aligned;
            param.normal_factor = normal_factor;
            param.qkv_precision = llmdnn::get_dt_from_str(qkv_precision_name);
            param.dst_precision = llmdnn::get_dt_from_str(dst_precision_name);
            param.max_seq_len = max_seq_len;
            param.rotary_emb_base = rotary_emb_base;
            param.rotary_pct = rotary_pct;
            param.use_position2d = use_position2d;
            if (param.qkv_precision == llmdnn::dnnl_data_type_undef)
                throw pybind11::type_error("Incorrect qkv type " + qkv_precision_name);
            if (param.dst_precision == llmdnn::dnnl_data_type_undef)
                throw pybind11::type_error("Incorrect dst type " + dst_precision_name);
            if (!self.create(param))
                throw pybind11::type_error("Incorrect param");
        },
        py::arg("num_heads"),
        py::arg("head_size"),
        py::arg("head_size_aligned"),
        py::arg("normal_factor"),
        py::arg("qkv_precision_name"),
        py::arg("dst_precision_name"),
        py::arg("max_seq_len"),
        py::arg("rotary_emb_base"),
        py::arg("rotary_pct"),
        py::arg("use_position2d") = false,
        R"(
            Create emb

            :param num_heads: heads number.
            :type num_heads: int
        )");
    cls.def("exec_position", [] (attn_gpt& self, const torch::Tensor& qkv, const torch::Tensor& layer_past_key_padded,
        const torch::Tensor& layer_past_value_padded, int64_t past_seq_len, const torch::Tensor& attn_mask, const torch::Tensor& position2d_ids) {
            // qkv: [batch, seq_len, (num_heads * 3 * head_size)]
            // layer_past_padded: [batch, num_attention_heads, MAX_SEQ_LEN, head_size_aligned]
            // past_seq_len: past_seq_len==layer_past.shape[-2]
            // attn_mask: [batch, 1, 1/query_seq_len, key_seq_len]
            // key/value: [batch, num_heads, query_seq_len+past_seq_len, head_size_aligned]
            AT_ASSERT(qkv.dim() == 3 && layer_past_key_padded.dim() == 4 && layer_past_value_padded.dim() == 4 && attn_mask.dim() == 4 &&
                qkv.size(0) == layer_past_key_padded.size(0) &&
                layer_past_key_padded.dim() == layer_past_value_padded.dim());
            auto batch = qkv.size(0);
            auto num_heads = layer_past_key_padded.size(1);
            auto query_seq_len = qkv.size(1);
            auto head_size = qkv.size(2) / 3 / num_heads;
            auto head_size_aligned = layer_past_key_padded.size(3);
            auto max_seq_len = layer_past_key_padded.size(2);
            AT_ASSERT(past_seq_len <= layer_past_key_padded.size(2) && head_size <= layer_past_key_padded.size(3) &&
                      query_seq_len <= layer_past_key_padded.size(2));

            attn_gpt::exec_param param;
            param.batch = batch;
            param.query_seq_len = query_seq_len;
            param.past_seq_len = past_seq_len;
            param.qkv = reinterpret_cast<uint8_t*>(qkv.data_ptr());
            param.layer_past_key_padded = reinterpret_cast<uint8_t**>(alloca(batch * sizeof(uint8_t*)));
            param.layer_past_value_padded = reinterpret_cast<uint8_t**>(alloca(batch * sizeof(uint8_t*)));
            for (int i = 0; i < batch; i++) {
                param.layer_past_key_padded[i] = reinterpret_cast<uint8_t*>(layer_past_key_padded[i].data_ptr());
                param.layer_past_value_padded[i] = reinterpret_cast<uint8_t*>(layer_past_value_padded[i].data_ptr());
            }
            param.position2d_ids = reinterpret_cast<int*>(position2d_ids.data_ptr());

            param.is_causal_in_attention = attn_mask.size(2) != 1;
            param.attention_mask = attn_mask.data_ptr<float>();
            param.head_stride_in_kv = max_seq_len * head_size_aligned;
            auto out = qkv.new_empty({batch, query_seq_len, num_heads * head_size});
            param.attn_output = reinterpret_cast<uint8_t*>(out.data_ptr());

            self.exec(param);

            // auto options = torch::TensorOptions().dtype(torch::kBFloat16);
            // auto query = torch::from_blob(param.query, {batch, num_heads, query_seq_len, head_size}, options);
            return out;
        },
        py::arg("qkv"),
        py::arg("layer_past_key_padded"),
        py::arg("layer_past_value_padded"),
        py::arg("past_seq_len"),
        py::arg("attn_mask"),
        py::arg("position2d_ids"),
        R"(
            exec emb

            :param num_heads: heads number.
            :type num_heads: int
        )");
}