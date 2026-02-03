// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef SNIPPETS_DEBUG_CAPS

#    pragma once

#    include <cstddef>
#    include <cstdlib>
#    include <memory>
#    include <sstream>
#    include <string>
#    include <typeinfo>

#    ifndef _WIN32
#        include <cxxabi.h>
#    endif

namespace ov::intel_cpu::snippets_common {

/**
 * @brief Get demangled type name of an emitter
 * @param emitter Pointer to the emitter object
 * @return Demangled type name string
 */
inline std::string get_emitter_type_name(const void* emitter) {
    std::string name = typeid(*static_cast<const char*>(emitter)).name();
#    ifndef _WIN32
    int status = 0;
    std::unique_ptr<char, void (*)(void*)> demangled_name(abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status),
                                                          std::free);
    if (status == 0 && demangled_name) {
        name = demangled_name.get();
    }
#    endif
    return name;
}

/**
 * @brief Generic template to get type name using typeid
 */
template <typename T>
std::string get_type_name(const T* obj) {
    std::string name = typeid(*obj).name();
#    ifndef _WIN32
    int status = 0;
    std::unique_ptr<char, void (*)(void*)> demangled_name(abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status),
                                                          std::free);
    if (status == 0 && demangled_name) {
        name = demangled_name.get();
    }
#    endif
    return name;
}

/**
 * @brief Get demangled type name of a jit_emitter
 * @param emitter Pointer to the emitter object
 * @return Demangled type name string
 */
template <typename JitEmitter>
std::string get_emitter_type_name(const JitEmitter* emitter) {
    return get_type_name(emitter);
}

/**
 * @brief Join elements of a container into a string
 * @param v Container with elements
 * @param sep Separator string (default: ", ")
 * @return Joined string
 */
template <typename T>
std::string join(const T& v, const std::string& sep = ", ") {
    std::ostringstream ss;
    size_t count = 0;
    for (const auto& x : v) {
        if (count++ > 0) {
            ss << sep;
        }
        ss << x;
    }
    return ss.str();
}

/**
 * @brief Convert vector to string with brackets
 * @param v Vector to convert
 * @return String representation in format "[ elem1, elem2, ... ]"
 */
template <typename T>
std::string vector_to_string(const T& v) {
    std::ostringstream os;
    os << "[ " << join(v) << " ]";
    return os.str();
}

/**
 * @brief Base class for emitter info structure
 * Provides common initialization and access methods
 */
struct jit_emitter_info_base {
    jit_emitter_info_base() = default;
    jit_emitter_info_base(const jit_emitter_info_base& rhs) = default;
    jit_emitter_info_base& operator=(const jit_emitter_info_base& rhs) = default;

    [[nodiscard]] const char* c_str() const {
        return str_.c_str();
    }

    [[nodiscard]] bool is_initialized() const {
        return is_initialized_;
    }

protected:
    std::string str_;
    bool is_initialized_ = false;
};

}  // namespace ov::intel_cpu::snippets_common

#endif  // SNIPPETS_DEBUG_CAPS
