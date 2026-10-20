// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "primitive.hpp"
namespace cldnn {
// Shared GPU implementation for the recurrent cache_r/cache_s operations introduced
// for Qwen3.5 in llama.cpp's OpenVINO backend (multi-slot inference and MTP rollback).
// history == 0 selects gather, positive history selects convolution cache, negative selects update.
struct recurrent_cache : public primitive_base<recurrent_cache> {
    CLDNN_DECLARE_PRIMITIVE(recurrent_cache)
    recurrent_cache() : primitive_base("", {}) {}
    recurrent_cache(const primitive_id& id, const std::vector<input_info>& inputs, int64_t history, int64_t key_size = 0, int64_t value_size = 0)
        : primitive_base(id, inputs),
          history(history),
          key_size(key_size),
          value_size(value_size) {
        num_outputs = history < 0 ? 1 : 2;
    }
    int64_t history = 0, key_size = 0, value_size = 0;
    size_t hash() const override {
        return hash_combine(hash_combine(hash_combine(primitive::hash(), history), key_size), value_size);
    }
    bool operator==(const primitive& rhs) const override {
        return compare_common_params(rhs) && history == downcast<const recurrent_cache>(rhs).history &&
               key_size == downcast<const recurrent_cache>(rhs).key_size && value_size == downcast<const recurrent_cache>(rhs).value_size;
    }
    void save(BinaryOutputBuffer& ob) const override {
        primitive_base::save(ob);
        ob << history << key_size << value_size;
    }
    void load(BinaryInputBuffer& ib) override {
        primitive_base::load(ib);
        ib >> history >> key_size >> value_size;
    }
};
}  // namespace cldnn
