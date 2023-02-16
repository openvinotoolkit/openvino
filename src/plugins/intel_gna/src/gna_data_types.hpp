// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "layers/gna_concat_layer.hpp"
#include "layers/gna_crop_layer.hpp"
#include "layers/gna_memory_layer.hpp"
#include "layers/gna_split_layer.hpp"
#include "memory/gna_memory.hpp"

struct TranspositionInfo {
    bool transpose;
    size_t num_transpose_rows;
    size_t num_transpose_columns;
};

using TranspositionInfoMap = std::map<std::string, std::vector<TranspositionInfo>>;

static inline bool FoundPartToTranspose(const std::vector<TranspositionInfo>& transpositionInfo) {
    auto partToTranspose =
        std::find_if(std::begin(transpositionInfo), std::end(transpositionInfo), [](const TranspositionInfo& infoPart) {
            return infoPart.transpose;
        });
    return partToTranspose != std::end(transpositionInfo);
}

namespace ov {
namespace intel_gna {

using gna_memory_type = memory::GNAMemoryInterface;
using gna_memory_float = memory::GNAMemory<memory::GNAFloatAllocator>;
using gna_memory_device = memory::GNAMemory<>;

using DnnComponentsForLayer = std::list<std::pair<std::string, intel_dnn_component_t>>;
using MemoryConnection = std::list<std::pair<std::string, GNAMemoryLayer>>;
using ConcatConnection = std::unordered_map<std::string, GNAConcatLayer>;
using SplitConnection = std::unordered_map<std::string, GNASplitLayer>;
using CropConnection = std::unordered_map<std::string, GNACropLayer>;
using ConstConnections = std::unordered_map<std::string, void*>;

}  // namespace intel_gna
}  // namespace ov
