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

#define FROM_IR_DIM(mem, idx)\
((mem->getTensorDesc().getDims().size() > (idx) - 1) ? mem->getTensorDesc().getDims()[mem->getTensorDesc().getDims().size() - (idx)] : 1)

struct TranspositionInfo {
    bool transpose;
    size_t num_transpose_rows;
    size_t num_transpose_columns;
};

using TranspositionInfoMap = std::map<std::string, std::vector<TranspositionInfo>>;

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
