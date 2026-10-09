// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <algorithm>
#include <common/utils.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

std::string jit_emitter_pretty_name(const std::string& pretty_func) {
    // Example:
    //      pretty_func := void ov::intel_cpu::jit_load_memory_emitter::emit_impl(const std::vector<size_t>& in, const
    //      std::vector<size_t>& out) const begin := -----------| end :=
    //      ---------------------------------------------------| result := ov::intel_cpu::jit_load_memory_emitter
    // Signatures:
    //      GCC:   void foo() [with T = {type}]
    //      clang: void foo() [T = {type}]
    //      MSVC:  void __cdecl foo<{type}>(void)
    auto parenthesis = pretty_func.find('(');
    if (any_of(parenthesis, std::string::npos, 0U)) {
        return pretty_func;
    }
    if (pretty_func[parenthesis - 1] == '>') {  // To cover template on MSVC
        parenthesis--;
        size_t counter = 1;
        while (counter != 0 && parenthesis > 0) {
            parenthesis--;
            if (pretty_func[parenthesis] == '>') {
                counter++;
            }
            if (pretty_func[parenthesis] == '<') {
                counter--;
            }
        }
    }
    auto end = pretty_func.substr(0, parenthesis).rfind("::");
    if (any_of(end, std::string::npos, 0U)) {
        return pretty_func;
    }

    auto begin = pretty_func.substr(0, end).rfind(' ');
    if (any_of(begin, std::string::npos, 0U)) {
        return pretty_func;
    }
    begin++;
    return end > begin ? pretty_func.substr(begin, end - begin) : pretty_func;
}

ov::element::Type get_arithmetic_binary_exec_precision(const std::shared_ptr<ov::Node>& n) {
    std::vector<ov::element::Type> input_precisions;
    for (const auto& input : n->inputs()) {
        input_precisions.push_back(input.get_source_output().get_element_type());
    }

    OPENVINO_ASSERT(std::all_of(input_precisions.begin(),
                                input_precisions.end(),
                                [&input_precisions](const ov::element::Type& precision) {
                                    return precision == input_precisions[0];
                                }),
                    "Binary Eltwise op has unequal input precisions");

    return input_precisions[0];
}

std::pair<int32_t, int32_t> get_clamp_min_max(double alpha, double beta, const ov::element::Type& exec_prc) {
    int32_t minimum = 0;
    int32_t maximum = 0;
    switch (exec_prc) {
    case ov::element::i32: {
        // Clamp alpha/beta to the int32_t range in double space before casting:
        // casting an out-of-range double to an integral type is undefined behavior.
        constexpr auto i32_min = static_cast<double>(std::numeric_limits<int32_t>::min());
        constexpr auto i32_max = static_cast<double>(std::numeric_limits<int32_t>::max());
        minimum = static_cast<int32_t>(std::clamp(alpha, i32_min, i32_max));
        maximum = static_cast<int32_t>(std::clamp(beta, i32_min, i32_max));
        break;
    }
    case ov::element::f32:
        minimum = dnnl::impl::float2int(static_cast<float>(alpha));
        maximum = dnnl::impl::float2int(static_cast<float>(beta));
        break;
    default:
        OPENVINO_THROW("Unsupported precision for Clamp min/max computation: ", exec_prc.to_string());
    }
    return {minimum, maximum};
}

}  // namespace ov::intel_cpu
