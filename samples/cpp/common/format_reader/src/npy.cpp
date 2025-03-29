// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <iostream>
#include <algorithm>

#include "npy.h"
// clang-format on

using namespace FormatReader;

NumpyArray::NumpyArray(const std::string& filename) {
    auto pos = filename.rfind('.');
    if (pos == std::string::npos)
        return;
    if (filename.substr(pos + 1) != "npy")
        return;

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return;
    }

    file.seekg(0, std::ios_base::end);
    if (!file.good()) {
        return;
    }
    auto full_file_size = static_cast<std::size_t>(file.tellg());
    file.seekg(0, std::ios_base::beg);

    std::string magic_string(6, ' ');
    file.read(&magic_string[0], magic_string.size());
    if (magic_string != "\x93NUMPY") {
        return;
    }

    file.ignore(2);
    unsigned short header_size;
    file.read((char*)&header_size, sizeof(header_size));

    std::string header(header_size, ' ');
    file.read(&header[0], header.size());

    int idx, from, to;

    // Verify fortran order is false
    const std::string fortran_key = "'fortran_order':";
    idx = header.find(fortran_key);
    if (idx == -1) {
        return;
    }

    from = header.find_last_of(' ', idx + fortran_key.size()) + 1;
    to = header.find(',', from);
    auto fortran_value = header.substr(from, to - from);
    if (fortran_value != "False") {
        return;
    }

    // Verify array shape matches the input's
    const std::string shape_key = "'shape':";
    idx = header.find(shape_key);
    if (idx == -1) {
        return;
    }

    from = header.find('(', idx + shape_key.size()) + 1;
    to = header.find(')', from);

    std::string shape_data = header.substr(from, to - from);

    if (!shape_data.empty()) {
        shape_data.erase(std::remove(shape_data.begin(), shape_data.end(), ','), shape_data.end());

        std::istringstream shape_data_stream(shape_data);
        size_t value;
        while (shape_data_stream >> value) {
            _shape.push_back(value);
        }
    }

    // Batch / Height / Width / Other dims
    // If batch is present, height and width are at least 1
    if (_shape.size()) {
        _height = _shape.size() >= 2 ? _shape.at(1) : 1;
        _width = _shape.size() >= 3 ? _shape.at(2) : 1;
    } else {
        _height = 0;
        _width = 0;
    }

    // Verify array data type matches input's
    std::string dataTypeKey = "'descr':";
    idx = header.find(dataTypeKey);
    if (idx == -1) {
        return;
    }

    from = header.find('\'', idx + dataTypeKey.size()) + 1;
    to = header.find('\'', from);
    type = header.substr(from, to - from);

    _size = full_file_size - static_cast<std::size_t>(file.tellg());

    _data.reset(new unsigned char[_size], std::default_delete<unsigned char[]>());
    for (size_t i = 0; i < _size; i++) {
        unsigned char buffer = 0;
        file.read(reinterpret_cast<char*>(&buffer), sizeof(buffer));
        _data.get()[i] = buffer;
    }
}
