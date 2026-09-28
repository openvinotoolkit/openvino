// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include <cstring>

#include "intel_npu/utils/zero/zero_init.hpp"
#include "intel_npu/utils/logger/logger.hpp"

namespace intel_npu {

void Plugin::schedule_builder_get_io_mapping(elf_binary& blob,
    io_map& inputs,
    io_map& outputs,
    parameter_map& parameters) const {
    auto io_text{blob};
    schedule_builder_proceed(io_text, ::intel_npu::BuilderOption::io_mapping, "");
    get_io_port_mapping(io_text, inputs, outputs, parameters);
}

void Plugin::schedule_builder_proceed(std::vector<elf_binary>& blobs,
                                  std::vector<BuilderOption> options,
                                  const std::vector<std::string>& option_param) const {
    auto& zeGraphExt{_backend->getInitStructs()};

    std::string option_parameter{};
    {
        auto option_str_iter{std::begin(option_param)};

        for (auto option : options) {
            option_parameter.append(builder_option_name[option]);
            option_parameter.append(";");

            if (option_str_iter->length()) {
                option_parameter.append(*option_str_iter);
                option_parameter.append(";");
            }

            option_str_iter++;
        }
    }

    std::vector<_ze_graph_desc_4_t> l0_descriptors{blobs.size()};

    {
        size_t blob_index{};
        _ze_graph_desc_4_t* prev_desc{};

        for (auto& desc : l0_descriptors) {
            desc.stype = ZE_STRUCTURE_TYPE_GRAPH_DESC_4;
            desc.format = ZE_GRAPH_FORMAT_NGRAPH_LITE;
            desc.inputSize = blobs[blob_index].size();
            desc.pInput = reinterpret_cast<uint8_t*>(blobs[blob_index].data());
            desc.flags = ZE_GRAPH_FLAG_NONE;
            desc.outputSize = 0ull;
            desc.pOutput = nullptr;
            desc.pBuildFlags = option_parameter.c_str();

            if (prev_desc) {
                prev_desc->pNext = reinterpret_cast<void*>(&desc);
            }

            prev_desc = &desc;
            blob_index++;
        }
    }

    auto result = zeGraphExt->getGraphDdiTable().pfnCreate4(zeGraphExt->getContext(),
                                                            zeGraphExt->getDevice(),
                                                            &l0_descriptors[0]);

    if (result == ZE_RESULT_SUCCESS) {
        blobs[0].resize(l0_descriptors[0].outputSize);
        memcpy_s(blobs[0].data(),
                 l0_descriptors[0].outputSize,
                 l0_descriptors[0].pOutput,
                 l0_descriptors[0].outputSize);
    } else {
        blobs.clear();
    }

    // Free heap allocated memory
    result = zeGraphExt->getGraphDdiTable().pfnCreate4(zeGraphExt->getContext(),
                                                       zeGraphExt->getDevice(),
                                                       &l0_descriptors[0]);
}

void Plugin::schedule_builder_proceed(elf_binary& blob, BuilderOption option, const std::string& option_param) const {
    auto& zeGraphExt{_backend->getInitStructs()};

    std::string option_parameter{builder_option_name[option]};
    option_parameter.append(";");
    option_parameter.append(option_param);

    _ze_graph_desc_4_t desc = {ZE_STRUCTURE_TYPE_GRAPH_DESC_4,
                               nullptr,
                               ZE_GRAPH_FORMAT_NGRAPH_LITE,
                               blob.size(),
                               reinterpret_cast<uint8_t*>(blob.data()),
                               option_parameter.c_str(),
                               ZE_GRAPH_FLAG_NONE,
                               0ull,
                               nullptr};

    _logger.debug("getGraphDescriptor - perform pfnCreate4");

    auto result = zeGraphExt->getGraphDdiTable().pfnCreate4(zeGraphExt->getContext(),
                                                            zeGraphExt->getDevice(),
                                                            &desc);

    if (result == ZE_RESULT_SUCCESS) {
        blob.resize(desc.outputSize);
        memcpy_s(blob.data(), desc.outputSize, desc.pOutput, desc.outputSize);
    } else {
        blob.clear();
    }

    // Free heap allocated memory
    result = zeGraphExt->getGraphDdiTable().pfnCreate4(zeGraphExt->getContext(),
                                                       zeGraphExt->getDevice(),
                                                       &desc);
}

}  // namespace intel_npu
