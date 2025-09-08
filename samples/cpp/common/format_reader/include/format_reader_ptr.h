// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief Implementation of smart pointer for Reader class
 * \file format_reader_ptr.h
 */
#pragma once

#include <functional>
#include <memory>

#include "format_reader.h"

namespace FormatReader {
class ReaderPtr {
public:
    explicit ReaderPtr(const char* imageName) : reader(CreateFormatReader(imageName)) {}
    /**
     * @brief dereference operator overload
     * @return Reader
     */
    Reader* operator->() const noexcept {
        return reader.get();
    }

    /**
     * @brief dereference operator overload
     * @return Reader
     */
    Reader* operator*() const noexcept {
        return reader.get();
    }

    Reader* get() {
        return reader.get();
    }

protected:
    std::unique_ptr<Reader> reader;
};
}  // namespace FormatReader
