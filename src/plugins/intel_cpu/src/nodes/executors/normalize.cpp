// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "normalize.hpp"
#include <common/primitive_hashing_utils.hpp>

namespace ov {
namespace intel_cpu {

size_t NormalizeKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    seed = hash_combine(seed, attrs.epsMode);
    seed = hash_combine(seed, attrs.across_spatial);
    seed = hash_combine(seed, attrs.cornerCase);
    seed = hash_combine(seed, attrs.eps);
    seed = hash_combine(seed, attrs.layout);
    seed = hash_combine(seed, attrs.input_prec.getPrecVal());
    seed = hash_combine(seed, attrs.output_prec.getPrecVal());

    seed = hash_combine(seed, get_attr_hash(*kernel_attrs.get()));
    seed = get_vector_hash(seed, attrs.vectorDims);
    return seed;
}

bool NormalizeKey::operator==(const NormalizeKey& rhs) const {
    return (attrs.epsMode == rhs.attrs.epsMode) && (attrs.across_spatial == rhs.attrs.across_spatial) &&
           (attrs.cornerCase == rhs.attrs.cornerCase) && (attrs.eps == rhs.attrs.eps) &&
           (attrs.layout == rhs.attrs.layout) && (attrs.input_prec == rhs.attrs.input_prec) &&
           (attrs.output_prec == rhs.attrs.output_prec) && (*kernel_attrs.get() == *(rhs.kernel_attrs.get())) &&
           (attrs.vectorDims == rhs.attrs.vectorDims);
}

NormalizeL2Executor::NormalizeL2Executor(const ExecutorContext::CPtr context) : implContext(context) {}

}   // namespace intel_cpu
}   // namespace ov