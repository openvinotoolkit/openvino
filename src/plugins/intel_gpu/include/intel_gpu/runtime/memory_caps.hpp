// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <ostream>

namespace cldnn {

enum class allocation_type {
    unknown,     // Not specified (i.e simple_attached_memory class).
    cl_mem,      // Use standard OpenCL cl_mem allocations.
    usm_host,    // Accessible by host and device. Not Migratable
    usm_shared,  // Accessible by host and device. Migrtable.
    usm_device,  // Accessible only by device. Not migratable.
    max_value,   // Used for data array size. Shall be last
};

inline std::ostream& operator<<(std::ostream& out, const allocation_type& alloc_type) {
    switch (alloc_type) {
        case allocation_type::cl_mem:     out << "cl_mem";     break;
        case allocation_type::usm_host:   out << "usm_host";   break;
        case allocation_type::usm_shared: out << "usm_shared"; break;
        case allocation_type::usm_device: out << "usm_device"; break;
        default: out << "unknown"; break;
    }

    return out;
}

class memory_capabilities {
public:
    memory_capabilities(std::vector<allocation_type> supported_allocation_types) : _caps(supported_allocation_types) {}
    bool supports_usm() const {
        return find_in_caps(allocation_type::usm_host) ||
               find_in_caps(allocation_type::usm_shared) ||
               find_in_caps(allocation_type::usm_device);
    }
    bool support_allocation_type(allocation_type type) const { return find_in_caps(type); }

    static bool is_usm_type(allocation_type type) {
        if (type == allocation_type::usm_host ||
            type == allocation_type::usm_shared ||
            type == allocation_type::usm_device)
            return true;
        return false;
    }

private:
    std::vector<allocation_type> _caps;

    bool find_in_caps(const allocation_type& type) const {
        return std::find_if(_caps.begin(), _caps.end(), [&](const allocation_type& t) { return t == type; }) != _caps.end();
    }
};


/// @brief Shared memory descriptor type.
enum class shared_mem_type {
    /// @brief Structure unitialized or contains no information.
    shared_mem_empty,

    /// @brief Structure describes shared CL buffer.
    shared_mem_buffer,

    /// @brief Structure describes shared CL image.
    shared_mem_image,

    /// @brief Structure describes shared VA/DXVA surface
    shared_mem_vasurface,

    /// @brief Structure describes shared D3D11 buffer
    shared_mem_dxbuffer,

    /// @brief Structure describes shared USM memory.
    shared_mem_usm
};

using shared_handle = void*;
using shared_surface = uint32_t;

/// @brief Low-level API handles required for using cldnn memory objects in external API calls.
struct shared_mem_params {
    shared_mem_type mem_type;     ///< shared buffer type
    shared_handle context;        ///< OpenCL context for external operations
    shared_handle user_device;    ///< DX/VA device for external operations
    shared_handle mem;            ///< memory object handle
#ifdef _WIN32
    shared_handle surface;        ///< VA/DXVA surface handle
#else
    shared_surface surface;
#endif
    uint32_t plane;               ///< shared surface plane
};

}  // namespace cldnn
