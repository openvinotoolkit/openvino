// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief The header file defines IMmap interface
 * 
 * @file ie_immap.hpp
 */
#pragma once

#include <details/ie_irelease.hpp>

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    using path_type = std::wstring;
#else
    using path_type = std::string;
#endif

namespace InferenceEngine {
/*
 * @brief This is a helper class to wrap memory mapped files
 */
class IMmap : public details::IRelease {
public:
    using Ptr = std::shared_ptr<IMmap>;

    /**
     * @brief Returns a pointer to mapped memory
     * @return A handle to the mapped memory or nullptr
     */
    void* data() noexcept {
        return _data;
    }

    /**
     * @brief Deletes current instance.
     */
    void Release() noexcept override {
        unmap();
        delete this;
    }

    /**
     * @return size of mapped file
     */
    size_t size() {
        return _size;
    }

protected:
    virtual ~IMmap() = default;

    /**
     * @brief Unmap mapped file.
     */
    virtual void map(const path_type& path, size_t size, size_t offset, LockOp lock) = 0;

    /**
     * @brief Unmap mapped file.
     */
    virtual void unmap() = 0;

protected:
    size_t _size;
    void* _data;
};

}  // namespace InferenceEngine
