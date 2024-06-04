// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <exception>
#include <fstream>
#include <numeric>
#include <vector>

/// \brief      Reads a binary file to a vector.
///
/// \param[in]  path  The path where the file is located.
///
/// \tparam     T     The type we want to interpret as the elements in binary file.
///
/// \return     Return vector of data read from input binary file.
///
template <typename T>
std::vector<T> read_binary_file(const std::string& path) {
    std::vector<T> file_content;
    std::ifstream inputs_fs{path, std::ios::in | std::ios::binary};
    if (!inputs_fs) {
        throw std::runtime_error("Failed to open the file: " + path);
    }

    inputs_fs.seekg(0, std::ios::end);
    auto size = inputs_fs.tellg();
    inputs_fs.seekg(0, std::ios::beg);
    if (size % sizeof(T) != 0) {
        throw std::runtime_error("Error reading binary file content: Input file size (in bytes) "
                                 "is not a multiple of requested data type size.");
    }
    file_content.resize(size / sizeof(T));
    inputs_fs.read(reinterpret_cast<char*>(file_content.data()), size);
    return file_content;
}

template <typename T = int>
std::vector<T> gen_range(const size_t elements, const T start = T{0}) {
    std::vector<T> range;
    range.resize(elements);
    std::iota(range.begin(), range.end(), start);

    return range;
}
