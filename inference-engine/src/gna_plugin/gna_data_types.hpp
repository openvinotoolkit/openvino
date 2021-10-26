// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>
#include <list>
#include <string>
#include <unordered_map>
#include "layers/gna_crop_layer.hpp"
#include "layers/gna_memory_layer.hpp"
#include "layers/gna_concat_layer.hpp"
#include "layers/gna_split_layer.hpp"
#include "gna_api_wrapper.hpp"
#include "memory/polymorph_allocator.hpp"
#include "memory/gna_memory.hpp"

struct TranspositionInfo {
    bool transpose;
    size_t num_transpose_rows;
    size_t num_transpose_columns;
};

using TranspositionInfoMap = std::map<std::string, std::vector<TranspositionInfo>>;

static inline bool FoundPartToTranspose(const std::vector<TranspositionInfo> &transpositionInfo) {
    auto partToTranspose = std::find_if(std::begin(transpositionInfo), std::end(transpositionInfo),
        [](const TranspositionInfo &infoPart) { return infoPart.transpose; });
    return partToTranspose != std::end(transpositionInfo);
}

namespace GNAPluginNS {
#if  GNA_LIB_VER == 2
    using dnn_ptr = std::shared_ptr<CPPWrapper<Gna2Model>>;
#else
    using dnn_ptr = std::shared_ptr<CPPWrapper<intel_nnet_type_t>>;
#endif
    using allocator_type = GNAPluginNS::memory::PolymorphAllocator<uint8_t>;
    using gna_memory_type = GNAPluginNS::memory::GNAMemory<allocator_type>;
    using DnnComponentsForLayer = std::list<std::pair<std::string, intel_dnn_component_t>>;
    using MemoryConnection = std::list<std::pair<std::string, GNAMemoryLayer>>;
    using ConcatConnection = std::unordered_map<std::string, GNAConcatLayer>;
    using SplitConnection = std::unordered_map<std::string, GNASplitLayer>;
    using CropConnection = std::unordered_map<std::string, GNACropLayer>;
    using ConstConnections = std::unordered_map<std::string, void*>;
}  // namespace GNAPluginNS
