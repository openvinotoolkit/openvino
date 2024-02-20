// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>
#include "buffer.hpp"
#include "helpers.hpp"
#include "kernel_selector_common.h"
#include "intel_gpu/runtime/kernel_args.hpp"

namespace cldnn {

template <typename BufferType>
class Serializer<BufferType, kernel_selector::clKernelData, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const kernel_selector::clKernelData& data) {
        const auto& params = data.params;
        buffer(params.workGroups.global, params.workGroups.local);
        buffer << params.arguments.size();
        for (const auto& arg : params.arguments) {
            buffer << make_data(&arg.t, sizeof(argument_desc::Types)) << arg.index;
        }
        buffer << params.scalars.size();
        for (const auto& scalar : params.scalars) {
            buffer << make_data(&scalar.t, sizeof(scalar_desc::Types)) << make_data(&scalar.v, sizeof(scalar_desc::ValueT));
        }
        buffer << params.layerID;
    }
};

template <typename BufferType>
class Serializer<BufferType, kernel_selector::clKernelData, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, kernel_selector::clKernelData& data) {
        auto& params = data.params;
        buffer(params.workGroups.global, params.workGroups.local);

        typename arguments_desc::size_type arguments_desc_size = 0UL;
        buffer >> arguments_desc_size;
        params.arguments.resize(arguments_desc_size);
        for (auto& arg : params.arguments) {
            buffer >> make_data(&arg.t, sizeof(argument_desc::Types)) >> arg.index;
        }

        typename scalars_desc::size_type scalars_desc_size = 0UL;
        buffer >> scalars_desc_size;
        params.scalars.resize(scalars_desc_size);
        for (auto& scalar : params.scalars) {
                buffer >> make_data(&scalar.t, sizeof(scalar_desc::Types)) >> make_data(&scalar.v, sizeof(scalar_desc::ValueT));
        }

        buffer >> params.layerID;
    }
};

}  // namespace cldnn
