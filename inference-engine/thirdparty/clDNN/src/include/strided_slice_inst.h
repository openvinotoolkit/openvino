/*
// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "api/strided_slice.hpp"
#include "primitive_inst.h"
#include "error_handler.h"
#include <string>
#include <vector>

namespace cldnn {

template <typename T, typename DT, typename = typename std::enable_if<std::is_convertible<DT, T>::value>::type>
std::vector<T>& pad_vector_to_size(std::vector<T>& data, size_t size, DT value) {
    for (size_t i = data.size(); i < size; ++i) {
        data.push_back(static_cast<T>(value));
    }
    return data;
}

template <typename T, typename MT>
std::vector<T>& vector_assign_if_not_mask(std::vector<T>& dst, const T& src, const std::vector<MT>& mask) {
    for (size_t i = 0; i < dst.size(); ++i) {
        if (!mask[i])
            dst[i] = src;
    }
    return dst;
}

template <typename T, typename MT>
std::vector<T>& vector_assign_if_not_mask(std::vector<T>& dst, const std::vector<T>& src, const std::vector<MT>& mask) {
    for (size_t i = 0; i < dst.size(); ++i) {
        if (!mask[i])
            dst[i] = src[i];
    }
    return dst;
}

inline format get_default_format_for_dim(size_t dimension) {
    format dimensionFormat = format::bfyx;
    switch (dimension) {
    case 1:
    case 2:
    case 3:
    case 4:
        dimensionFormat = format::bfyx;
        break;
    case 5:
        dimensionFormat = format::bfzyx;
        break;
    case 6:
        dimensionFormat = format::bfwzyx;
        break;
    default:
        CLDNN_ERROR_MESSAGE("Function get_default_format_for_dim", "Unsupported dimension number: " + std::to_string(dimension));
    }
    return dimensionFormat;
}

template <>
struct typed_program_node<strided_slice> : public typed_program_node_base<strided_slice> {
    using parent = typed_program_node_base<strided_slice>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
};

using strided_slice_node = typed_program_node<strided_slice>;

template <>
class typed_primitive_inst<strided_slice> : public typed_primitive_inst_base<strided_slice> {
    using parent = typed_primitive_inst_base<strided_slice>;

public:
    static layout calc_output_layout(strided_slice_node const& node);
    static std::string to_string(strided_slice_node const& node);

public:
    typed_primitive_inst(network_impl& network, strided_slice_node const& desc);
};

using strided_slice_inst = typed_primitive_inst<strided_slice>;
}  // namespace cldnn
