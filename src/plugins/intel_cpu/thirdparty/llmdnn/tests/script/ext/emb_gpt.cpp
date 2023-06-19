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
#include "test_common.hpp"

using namespace torch::indexing;

void regclass_emb_gpt(pybind11::module m) {
    py::class_<llmdnn::emb_gpt, std::shared_ptr<llmdnn::emb_gpt>> cls(m, "emb_gpt");
    cls.def(py::init<>());
    cls.def("create", [] (llmdnn::emb_gpt& self,
        const size_t num_heads,
        const size_t head_size,
        const size_t head_size_aligned,
        const std::string qkv_precision_name,
        const std::string dst_precision_name,
        const size_t max_seq_len,
        const size_t rotary_emb_base,
        float rotary_pct) {
            llmdnn::emb_gpt::create_param param;
            param.num_heads = num_heads;
            param.head_size = head_size;
            param.head_size_aligned = head_size_aligned;
            param.qkv_precision = llmdnn::get_dt_from_str(qkv_precision_name);
            param.dst_precision = llmdnn::get_dt_from_str(dst_precision_name);
            param.max_seq_len = max_seq_len;
            param.rotary_emb_base = rotary_emb_base;
            param.rotary_pct = rotary_pct;
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
        py::arg("qkv_precision_name"),
        py::arg("dst_precision_name"),
        py::arg("max_seq_len"),
        py::arg("rotary_emb_base"),
        py::arg("rotary_pct"),
        R"(
            Create emb

            :param num_heads: heads number.
            :type num_heads: int
        )");
    // torch::List
    cls.def("exec", [] (llmdnn::emb_gpt& self, const torch::Tensor& qkv, const torch::Tensor& layer_past_key_padded,
        const torch::Tensor& layer_past_value_padded, const torch::Tensor& query_padded, size_t past_seq_len) {
            // qkv: [batch, seq_len, (num_heads * 3 * head_size)]
            // layer_past_padded: [batch, num_attention_heads, MAX_SEQ_LEN, head_size_aligned]
            // past_seq_len: past_seq_len==layer_past.shape[-2]
            // query_padded: [batch, num_heads, query_seq_len, head_size_aligned]
            // key/value: [batch, num_heads, query_seq_len+past_seq_len, head_size_aligned]
            AT_ASSERT(qkv.dim() == 3 && layer_past_key_padded.dim() == 4 && layer_past_value_padded.dim() == 4 &&
                qkv.size(0) == layer_past_key_padded.size(0) &&
                layer_past_key_padded.dim() == layer_past_value_padded.dim());
            AT_ASSERT(query_padded.dim() == 4 && query_padded.size(0) == qkv.size(0) &&
                query_padded.size(1) == layer_past_key_padded.size(1) && query_padded.size(2) == qkv.size(1) &&
                query_padded.size(3) == layer_past_key_padded.size(3));
            auto batch = qkv.size(0);
            auto num_heads = layer_past_key_padded.size(1);
            auto query_seq_len = qkv.size(1);
            auto head_size = qkv.size(2) / 3 / num_heads;
            AT_ASSERT(past_seq_len <= layer_past_key_padded.size(2) && head_size <= layer_past_key_padded.size(3) &&
                      query_seq_len <= layer_past_key_padded.size(2));

            llmdnn::emb_gpt::exec_param param;
            param.batch = batch;
            param.query_seq_len = query_seq_len;
            param.past_seq_len = past_seq_len;
            param.qkv = reinterpret_cast<uint8_t*>(qkv.data_ptr());
            param.query_dst = reinterpret_cast<uint8_t*>(query_padded.data_ptr());
            param.layer_past_key_padded = reinterpret_cast<uint8_t**>(alloca(batch * sizeof(uint8_t*)));
            param.layer_past_value_padded = reinterpret_cast<uint8_t**>(alloca(batch * sizeof(uint8_t*)));
            for (int i = 0; i < batch; i++) {
                param.layer_past_key_padded[i] = reinterpret_cast<uint8_t*>(layer_past_key_padded[i].data_ptr());
                param.layer_past_value_padded[i] = reinterpret_cast<uint8_t*>(layer_past_value_padded[i].data_ptr());
            }

            self.exec(param);

            // auto options = torch::TensorOptions().dtype(torch::kBFloat16);
            // auto query = torch::from_blob(param.query, {batch, num_heads, query_seq_len, head_size}, options);
        },
        py::arg("qkv"),
        py::arg("layer_past_key_padded"),
        py::arg("layer_past_value_padded"),
        py::arg("query_padded"),
        py::arg("past_seq_len"),
        R"(
            exec emb

            :param num_heads: heads number.
            :type num_heads: int
        )");
}