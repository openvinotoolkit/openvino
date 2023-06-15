// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <torch/extension.h>
#include <memory>
#include "alloca.h"
#include "module.hpp"
#include "common/utility.hpp"
#include "utility_amx.hpp"
#include "llm_mha_gpt.hpp"
#include "test_common.hpp"

void regclass_mha_gpt(pybind11::module m) {
    py::class_<llmdnn::mha_gpt, std::shared_ptr<llmdnn::mha_gpt>> cls(m, "mha_gpt");
    cls.def(py::init<>());
    cls.def("create", [] (llmdnn::mha_gpt& self,
        const size_t num_heads,
        const size_t head_size,
        const size_t head_size_aligned,
        const float normal_factor,
        const std::string qkv_precision_name,
        const std::string dst_precision_name,
        const size_t max_seq_len) {
            llmdnn::mha_gpt::create_param param;
            param.num_heads = num_heads;
            param.head_size = head_size;
            param.head_size_aligned = head_size_aligned;
            param.normal_factor = normal_factor;
            param.qkv_precision = llmdnn::get_dt_from_str(qkv_precision_name);
            param.dst_precision = llmdnn::get_dt_from_str(dst_precision_name);
            param.max_seq_len = max_seq_len;
            if (param.qkv_precision == llmdnn::dnnl_data_type_undef)
                throw pybind11::type_error("Incorrect qkv type " + qkv_precision_name);
            if (param.dst_precision == llmdnn::dnnl_data_type_undef)
                throw pybind11::type_error("Incorrect dst type " + dst_precision_name);
            self.create(param);
        },
        py::arg("num_heads"),
        py::arg("head_size"),
        py::arg("head_size_aligned"),
        py::arg("normal_factor"),
        py::arg("qkv_precision_name"),
        py::arg("dst_precision_name"),
        py::arg("max_seq_len"),
        R"(
            Create mha

            :param num_heads: heads number.
            :type num_heads: int
        )");
    cls.def("exec", [] (llmdnn::mha_gpt& self, torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor attn_mask) {
            // q: [batch, num_heads, query_seq_len, head_size]
            // k: [batch, num_heads, key_seq_len, head_size]
            // v: [batch, num_heads, value_seq_len, head_size]
            // attn_mask: [batch, MAX_POSITION_EMBEDDINGS]
            // out: [batch, query_seq_len, num_heads * head_size]
            AT_ASSERT(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && attn_mask.dim() == 2);
            auto batch = q.size(0);
            auto num_heads = q.size(1);
            auto query_seq_len = q.size(2);
            auto head_size = q.size(3);
            auto key_seq_len = k.size(2);
            auto attn_len = attn_mask.size(1);
            AT_ASSERT(key_seq_len == v.size(2) && key_seq_len == attn_len &&
                    batch == k.size(0) && batch == v.size(0) && 1 == attn_mask.size(0) &&
                    num_heads == k.size(1) && num_heads == v.size(1) &&
                    head_size == k.size(3) && head_size == v.size(3));

            auto out = q.new_empty({batch, query_seq_len, num_heads * head_size});
            llmdnn::mha_gpt::exec_param param;
            param.batch = batch;
            param.query_seq_len = query_seq_len;
            param.key_seq_len = key_seq_len;
            param.q = reinterpret_cast<uint8_t*>(q.data_ptr());
            param.attn_output = reinterpret_cast<uint8_t*>(out.data_ptr());
            param.head_stride_in_kv = key_seq_len * head_size;
            param.k = reinterpret_cast<uint8_t**>(alloca(batch * sizeof(uint8_t*)));
            param.v = reinterpret_cast<uint8_t**>(alloca(batch * sizeof(uint8_t*)));
            param.attention_mask = reinterpret_cast<float**>(alloca(batch * sizeof(float*)));
            for (int i = 0; i < batch; i++) {
                param.k[i] = reinterpret_cast<uint8_t*>(k.data_ptr()) + i * num_heads * key_seq_len * head_size * sizeof(ov::bfloat16);
                param.v[i] = reinterpret_cast<uint8_t*>(v.data_ptr()) + i * num_heads * key_seq_len * head_size * sizeof(ov::bfloat16);
                param.attention_mask[i] = attn_mask.data_ptr<float>();
            }

            self.exec(param);
            return out;
        },
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("attn_mask"),
        R"(
            exec mha

            :param num_heads: heads number.
            :type num_heads: int
        )");
}