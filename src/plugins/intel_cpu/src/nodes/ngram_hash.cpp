// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngram_hash.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <vector>

#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "shape_inference/custom/ngram_hash.hpp"
#include "transformations/cpu_opset/common/op/ngram_hash.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

bool NgramHash::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::as_type_ptr<const NgramHashNode>(op)) {
            errorMessage = "Only NgramHash from CPU internal opset is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

NgramHash::NgramHash(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgramHashShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto ngram_hash = ov::as_type_ptr<const NgramHashNode>(op);
    for (auto m : ngram_hash->get_multipliers()) {
        m_multipliers.push_back(static_cast<uint64_t>(m));
    }
    m_moduli = ngram_hash->get_moduli();
    m_offsets = ngram_hash->get_offsets();
}

void NgramHash::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }
    m_precision = getOriginalInputPrecisionAtPort(0);
    if (none_of(m_precision, ov::element::i32, ov::element::i64)) {
        m_precision = ov::element::i32;
    }
    std::vector<PortConfigurator> inputs(getOriginalInputsNumber(), {LayoutType::ncsp, m_precision});
    addSupportedPrimDesc(inputs, {{LayoutType::ncsp, m_precision}}, impl_desc_type::ref_any);
}

template <typename T>
void NgramHash::executeImpl() {
    const size_t num_inputs = m_multipliers.size();
    const size_t heads = m_moduli.size();
    std::vector<const T*> tokens(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        tokens[i] = getSrcDataAtPortAs<const T>(i);
    }
    auto* dst = getDstDataAtPortAs<T>(0);
    const size_t count = ov::shape_size(getSrcMemoryAtPort(0)->getStaticDims());

    // Same semantics as the traced int64 graph: wrapping int64 multiply/xor, then floor_mod (sign of divisor).
    auto hash_range = [&](size_t begin, size_t end) {
        for (size_t t = begin; t < end; ++t) {
            uint64_t mixed = 0;
            for (size_t i = 0; i < num_inputs; ++i) {
                mixed ^= static_cast<uint64_t>(static_cast<int64_t>(tokens[i][t])) * m_multipliers[i];
            }
            const auto value = static_cast<int64_t>(mixed);
            T* out = dst + t * heads;
            for (size_t h = 0; h < heads; ++h) {
                int64_t r = value % m_moduli[h];
                if (r < 0) {
                    r += m_moduli[h];
                }
                out[h] = static_cast<T>(r + m_offsets[h]);
            }
        }
    };

    constexpr size_t block = 256;
    const size_t num_blocks = (count + block - 1) / block;
    if (num_blocks <= 1) {
        hash_range(0, count);
        return;
    }
    context->getCpuParallel()->parallel_for(num_blocks, [&](size_t b) {
        hash_range(b * block, std::min(count, (b + 1) * block));
    });
}

void NgramHash::execute([[maybe_unused]] const dnnl::stream& strm) {
    if (m_precision == ov::element::i64) {
        executeImpl<int64_t>();
    } else {
        executeImpl<int32_t>();
    }
}

void NgramHash::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool NgramHash::created() const {
    return getType() == Type::NgramHash;
}

}  // namespace ov::intel_cpu::node
