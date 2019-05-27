// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/backend/backend.hpp>

#include <climits>
#include <cstring>

#include <string>
#include <memory>
#include <list>
#include <vector>
#include <array>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <fstream>
#include <utility>
#include <algorithm>
#include <map>
#include <streambuf>
#include <tuple>
#include <sstream>
#include <iomanip>
#include <atomic>

#include <precision_utils.h>
#include <details/caseless.hpp>
#include <graph_tools.hpp>
#include <description_buffer.hpp>
#include <xml_parse_utils.h>

#include <vpu/parsed_config.hpp>
#include <vpu/compile_env.hpp>
#include <vpu/backend/blob_format.hpp>
#include <vpu/utils/auto_scope.hpp>
#include <vpu/utils/dot_io.hpp>
#include <vpu/utils/file_system.hpp>
#include <vpu/utils/numeric.hpp>

namespace vpu {

void BackEnd::serialize(
        const Model::Ptr& model,
        std::vector<char>& blob,
        std::pair<char*, size_t>& blobHeader,
        int& numActiveStages) {
    VPU_PROFILE(serialize);

    const auto& env = CompileEnv::get();

    auto batchSize = model->batchSize();
    auto usedMemory = model->attrs().get<UsedMemory>("usedMemory");

    //
    // Remove special stages from the stages list
    //

    bool hasHwStage = false;
    bool hasShaveStage = false;
    bool hasDmaStage = false;

    StageVector execStages;
    execStages.reserve(model->numStages());

    for (const auto& stage : model->getStages()) {
        if (stage->category() == StageCategory::Special) {
            continue;
        }

        if (stage->category() == StageCategory::HW) {
            hasHwStage = true;
        } else if (stage->category() == StageCategory::SHAVE) {
            hasShaveStage = true;
        } else if (stage->category() == StageCategory::DMA) {
            hasDmaStage = true;
        }

        execStages.emplace_back(stage);
    }

    numActiveStages = execStages.size();

    //
    // I/O info sections
    //

    int numInputs = 0;
    BlobSerializer inputInfoSerializer;
    for (const auto& data : model->datas()) {
        if (data->usage() != DataUsage::Input) {
            continue;
        }

        IE_ASSERT(data->producerEdge() == nullptr);
        IE_ASSERT(data->parentDataEdge() == nullptr);
        IE_ASSERT(data->numConsumers() != 0);

        IE_ASSERT(!data->attrs().has("ioIdx"));
        data->attrs().set("ioIdx", numInputs);

        data->serializeIOInfo(inputInfoSerializer);

        ++numInputs;
    }

    int numOutputs = 0;
    BlobSerializer outputInfoSerializer;
    for (const auto& data : model->datas()) {
        if (data->usage() != DataUsage::Output) {
            continue;
        }

        IE_ASSERT(data->producerEdge() != nullptr);
        IE_ASSERT(data->parentDataEdge() == nullptr);

        IE_ASSERT(!data->attrs().has("ioIdx"));
        data->attrs().set("ioIdx", numOutputs);

        data->serializeIOInfo(outputInfoSerializer);

        ++numOutputs;
    }

    //
    // Stages section
    //


    BlobSerializer stagesSerializer;
    for (const auto& stage : execStages) {
        stage->serialize(stagesSerializer);
    }

    //
    // Elf header
    //

    ElfN_Ehdr elfHdr = {};
    elfHdr.e_ident[0] = 0x7f;
    elfHdr.e_ident[1] = 'e';
    elfHdr.e_ident[2] = 'l';
    elfHdr.e_ident[3] = 'f';
    for (int i = 4; i < 16; i++) {
        elfHdr.e_ident[i] = 0;
    }
    elfHdr.e_type = 1;
    elfHdr.e_machine = 2;
    elfHdr.e_version = 2;
    elfHdr.e_entry = 0;
    elfHdr.e_phoff = 0;
    elfHdr.e_shoff = 0;
    elfHdr.e_ehsize = 8 * sizeof(elfHdr);

    //
    // Blob header
    //

    auto hdrSize = alignVal<int>(sizeof(ElfN_Ehdr) + sizeof(mv_blob_header), 64);
    auto inputInfoSecSize = alignVal(inputInfoSerializer.size(), 64);
    auto outputInfoSecSize = alignVal(outputInfoSerializer.size(), 64);
    auto stagesSecSize = alignVal(stagesSerializer.size(), 64);
    auto constDataSecSize = alignVal(usedMemory.blob, 64);

    mv_blob_header blobHdr = {};
    blobHdr.magic_number = BLOB_MAGIC_NUMBER;
    blobHdr.file_size = checked_cast<uint32_t>(hdrSize + inputInfoSecSize + outputInfoSecSize + stagesSecSize + constDataSecSize);
    blobHdr.blob_ver_major = BLOB_VERSION_MAJOR;
    blobHdr.blob_ver_minor = BLOB_VERSION_MINOR;
    blobHdr.inputs_count = checked_cast<uint32_t>(numInputs);
    blobHdr.outputs_count = checked_cast<uint32_t>(numOutputs);
    blobHdr.stages_count = checked_cast<uint32_t>(execStages.size());
    blobHdr.inputs_size = checked_cast<uint32_t>(usedMemory.input);
    blobHdr.outputs_size = checked_cast<uint32_t>(usedMemory.output);
    blobHdr.batch_size = checked_cast<uint32_t>(batchSize);
    blobHdr.bss_mem_size = checked_cast<uint32_t>(usedMemory.BSS);
    blobHdr.number_of_cmx_slices = checked_cast<uint32_t>(env.resources.numCMXSlices);
    blobHdr.number_of_shaves = checked_cast<uint32_t>(env.resources.numSHAVEs);
    blobHdr.has_hw_stage = checked_cast<uint32_t>(hasHwStage);
    blobHdr.has_shave_stage = checked_cast<uint32_t>(hasShaveStage);
    blobHdr.has_dma_stage = checked_cast<uint32_t>(hasDmaStage);
    blobHdr.input_info_section_offset = checked_cast<uint32_t>(hdrSize);
    blobHdr.output_info_section_offset = checked_cast<uint32_t>(blobHdr.input_info_section_offset + inputInfoSecSize);
    blobHdr.stage_section_offset = checked_cast<uint32_t>(blobHdr.output_info_section_offset + outputInfoSecSize);
    blobHdr.const_data_section_offset = checked_cast<uint32_t>(blobHdr.stage_section_offset + stagesSecSize);

    //
    // Generate fathom blob
    //

    blob.clear();
    blob.resize(blobHdr.file_size, 0);

    std::copy_n(&elfHdr, 1, reinterpret_cast<ElfN_Ehdr*>(blob.data()));
    std::copy_n(&blobHdr, 1, reinterpret_cast<mv_blob_header*>(blob.data() + sizeof(elfHdr)));
    std::copy_n(inputInfoSerializer.data(), inputInfoSerializer.size(), blob.data() + blobHdr.input_info_section_offset);
    std::copy_n(outputInfoSerializer.data(), outputInfoSerializer.size(), blob.data() + blobHdr.output_info_section_offset);
    std::copy_n(stagesSerializer.data(), stagesSerializer.size(), blob.data() + blobHdr.stage_section_offset);

    for (const auto& data : model->datas()) {
        if (data->usage() != DataUsage::Const) {
            continue;
        }

        IE_ASSERT(data->producerEdge() == nullptr);
        IE_ASSERT(data->parentDataEdge() == nullptr);
        IE_ASSERT(data->numConsumers() != 0);
        IE_ASSERT(data->location() == DataLocation::Blob);

        auto content = data->content();
        IE_ASSERT(content != nullptr);

        std::copy_n(content->get<uint8_t>(), data->totalByteSize(), blob.data() + blobHdr.const_data_section_offset + data->memoryOffset());
    }

    //
    // Blob header spec begin containing elf header and blobHeader
    //

    blobHeader.first = blob.data();
    blobHeader.second = sizeof(ElfN_Ehdr) + sizeof(mv_blob_header);

    env.log->info("blobSize=%d", sizeof(char) * blob.size());
}

}  // namespace vpu
