// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(NGRAPH_LEGACY_HEADER_INCLUDED)
#    define NGRAPH_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <cstddef>

#include "ngraph/runtime/aligned_buffer.hpp"

namespace ngraph {
namespace runtime {
NGRAPH_SUPPRESS_DEPRECATED_START
/// \brief SharedBuffer class to store pointer to pre-acclocated buffer.
template <typename T>
class NGRAPH_API_DEPRECATED SharedBuffer : public ngraph::runtime::AlignedBuffer {
public:
    SharedBuffer(char* data, size_t size, const T& shared_object) : _shared_object(shared_object) {
        m_allocated_buffer = data;
        m_aligned_buffer = data;
        m_byte_size = size;
    }

    virtual ~SharedBuffer() {
        m_aligned_buffer = nullptr;
        m_allocated_buffer = nullptr;
        m_byte_size = 0;
    }

private:
    T _shared_object;
};
NGRAPH_SUPPRESS_DEPRECATED_END
}  // namespace runtime
}  // namespace ngraph
