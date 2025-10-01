// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "llm_compiled_model_utils.hpp"
#include "logging.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace npuw {
namespace util {

bool is_set(const std::size_t sub_idx,
            const std::string& opt,
            const std::size_t real_idx = SIZE_MAX,
            const std::size_t end_idx = SIZE_MAX);

// Every great project has its own string class...
// NB: Newer C++ standards would allow to use string views or smt
ov::Tensor tensor_from_const(const std::shared_ptr<ov::Node>& node);

// In case of working with memory which will be detached later (Constant will be freed),
// we need to explicitly create a tensor which owns the memory during the execution.
ov::Tensor copy_tensor_from_const(const std::shared_ptr<ov::Node>& node);

bool starts_with(const std::string& str, const std::string& prefix);

std::string fmt(std::size_t number, std::size_t total);

struct UnpackOptions {
    bool bUseOvParallelFor;
    size_t nPartitions;  // if 0 we use 64 elements step in parallel for, otherwise  target workload is dynamically
                         // calculated
    bool bStrictPartitioning;  // cannot reduce partitions in favor of speed
    explicit UnpackOptions(bool useParallelFor, size_t nPartitions, bool bStrictPartitioning)
        : bUseOvParallelFor(useParallelFor),
          nPartitions(nPartitions),
          bStrictPartitioning(bStrictPartitioning) {}
};

void unpack(const ov::SoPtr<ov::ITensor>& from,
            const ov::SoPtr<ov::ITensor>& to,
            const UnpackOptions& unpack_options = UnpackOptions{true, 16, false});

void unpack(const ov::SoPtr<ov::ITensor>& from,
            const ov::SoPtr<ov::ITensor>& scale,
            const ov::SoPtr<ov::ITensor>& to,
            const UnpackOptions& unpack_options = UnpackOptions{true, 16, false});

void unpack(const ov::SoPtr<ov::ITensor>& from,
            const ov::SoPtr<ov::ITensor>& zerop,
            const ov::SoPtr<ov::ITensor>& scale,
            const ov::SoPtr<ov::ITensor>& to,
            const UnpackOptions& unpack_options = UnpackOptions{true, 16, false});

void gather(const ov::SoPtr<ov::ITensor>& src, const ov::SoPtr<ov::ITensor>& idx, const ov::SoPtr<ov::ITensor>& dst);
void gather_cb4(const ov::SoPtr<ov::ITensor>& src,
                const ov::SoPtr<ov::ITensor>& idx,
                const ov::SoPtr<ov::ITensor>& dst);

using View = std::vector<std::size_t>;
ov::SoPtr<ov::ITensor> view(const ov::SoPtr<ov::ITensor>& src, const View& from, const View& to);

ov::SoPtr<ov::ITensor> view(const ov::SoPtr<ov::ITensor>& src, std::size_t dim, std::size_t offset, std::size_t len);

void to_f32(const ov::Tensor& in, ov::Tensor& out);
ov::Tensor to_f16(const ov::Tensor& t);
ov::Tensor transpose(const ov::Tensor& t);
ov::Tensor permute(const ov::Tensor& t, const std::vector<std::size_t>& axes);
ov::Tensor concat(const std::vector<ov::Tensor>& tt, std::size_t axis);

// Start is inclusive, end is exclusive
using range_1d = std::pair<std::size_t, std::size_t>;
range_1d validMaskRange(const ov::SoPtr<ov::ITensor>& t);

namespace at {
template <class M_>
struct Impl {
    using M = typename std::decay<M_>::type;
    using V = typename M::mapped_type;

    M* m = nullptr;
    explicit Impl(M* pM) : m(pM) {}

    template <typename K>
    V& at(const K& k) {
        const auto iter = m->find(k);
        if (iter == m->end()) {
            std::stringstream ss;
            ss << "Key " << k << " is not found in a map of type " << typeid(m).name();
            const auto msg = ss.str();
            LOG_ERROR(msg);
            throw std::out_of_range(msg);
        }
        return iter->second;
    }

    template <typename K>
    V& at_or_at(const K& k1, const K& k2) {
        const auto iter = m->find(k1);
        if (iter == m->end()) {
            return at(k2);
        }
        return iter->second;
    }

    template <typename K>
    V& at_or_at_or_at(const K& k1, const K& k2, const K& k3) {
        const auto iter = m->find(k1);
        if (iter == m->end()) {
            return at_or_at(k2, k3);
        }
        return iter->second;
    }

    template <typename K>
    const V& at(const K& k) const {
        return const_cast<Impl*>(this)->at(k);
    }

    template <typename K>
    const V& at_or_at(const K& k1, const K& k2) const {
        return const_cast<Impl*>(this)->at_or_at(k1, k2);
    }

    template <typename K>
    const V& at_or_at_or_at(const K& k1, const K& k2, const K& k3) const {
        return const_cast<Impl*>(this)->at_or_at_or_at(k1, k2, k3);
    }
};

template <typename M>
Impl<M> _(M* pM) {
    return Impl<M>(pM);
}

template <typename M>
Impl<M> _(M&& m) {
    return Impl<M>(&m);
}

template <typename M>
Impl<M> _(std::shared_ptr<M> pM) {
    return Impl<M>(pM.get());
}

}  // namespace at

// Written here to be a drop-in replacement for ov::parallel_for for the debug purposes
template <typename F>
void non_parallel_for(std::size_t count, F&& f) {
    for (std::size_t idx = 0u; idx < count; idx++) {
        f(idx);
    }
}

template <class CountedType>
struct Unique {
    static std::string name() {
        static std::size_t counter = 0u;
        return std::string(CountedType::name) + "_" + std::to_string(counter++);
    }
};

using TensorPtr = ov::SoPtr<ov::ITensor>;
TensorPtr allocMem(const ov::element::Type type,
                   const ov::Shape& shape,
                   const std::string& device,
                   const std::shared_ptr<const ov::IPlugin>& plugin);

bool matchStringWithLoRAPattern(const std::string& input, const std::string& pattern_suffix);

bool matchLoRAMatMulAString(const std::string& input);

bool matchLoRAMatMulBString(const std::string& input);

bool matchLoRAMatMulAlphaString(const std::string& input);

template <typename T>
void fill_tensor(ov::SoPtr<ov::ITensor> tensor, T fill_val, size_t offset = 0u) {
    T* tensor_data = tensor->data<T>();
    std::fill(tensor_data + offset, tensor_data + tensor->get_size(), fill_val);
}

void fill_tensor_bytes(ov::SoPtr<ov::ITensor> tensor, uint8_t fill_val);

bool isPastKeyValuesKey(const std::string& str);

bool isPastKeyValuesValue(const std::string& str);

void copy_by_planes(ov::SoPtr<ov::ITensor> src_tensor, ov::SoPtr<ov::ITensor> dst_tensor);

void copy_columns_by_row_chunks(ov::SoPtr<ov::ITensor> src, ov::SoPtr<ov::ITensor>& dst);

void copy_tensor_by_dim(ov::SoPtr<ov::ITensor> src_tensor, ov::SoPtr<ov::ITensor> dst_tensor, uint32_t kv_dim);

}  // namespace util
}  // namespace npuw
}  // namespace ov
