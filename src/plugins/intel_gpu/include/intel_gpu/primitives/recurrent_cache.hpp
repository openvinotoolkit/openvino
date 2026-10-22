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
    recurrent_cache(const primitive_id &id, const std::vector<input_info> &inputs, int64_t history,
                    int64_t key_size = 0, int64_t value_size = 0, bool prepare_cache = true)
        : primitive_base(id, inputs), history(history), key_size(key_size), value_size(value_size),
          prepare_cache(prepare_cache) {
        num_outputs = history < 0 || !prepare_cache ? 1 : 2;
    }
    int64_t history = 0, key_size = 0, value_size = 0;
    bool prepare_cache = true;
    size_t hash() const override {
        return hash_combine(hash_combine(hash_combine(hash_combine(primitive::hash(), history), key_size), value_size),
                            prepare_cache);
    }
    bool operator==(const primitive& rhs) const override {
        return compare_common_params(rhs) && history == downcast<const recurrent_cache>(rhs).history &&
               key_size == downcast<const recurrent_cache>(rhs).key_size &&
               value_size == downcast<const recurrent_cache>(rhs).value_size &&
               prepare_cache == downcast<const recurrent_cache>(rhs).prepare_cache;
    }
    void save(BinaryOutputBuffer& ob) const override {
        primitive_base::save(ob);
        ob << history << key_size << value_size << prepare_cache;
    }
    void load(BinaryInputBuffer& ib) override {
        primitive_base::load(ib);
        ib >> history >> key_size >> value_size >> prepare_cache;
    }
};
}  // namespace cldnn
