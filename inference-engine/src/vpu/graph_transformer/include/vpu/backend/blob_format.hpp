// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <cstddef>

#include <vpu/utils/extra.hpp>

namespace vpu {

// It is one of float NaN and prime number
const uint32_t STAGE_BORDER_SYMBOL = 0x7f83ff19;

const uint32_t EI_NIDENT = 16;

VPU_PACKED(ElfN_Ehdr {
    uint8_t  e_ident[EI_NIDENT];
    uint16_t e_type;
    uint16_t e_machine;
    uint32_t e_version;
    uint32_t e_entry;
    uint32_t e_phoff;
    uint32_t e_shoff;
    uint32_t e_flags;
    uint16_t e_ehsize;
    uint16_t e_phentsize;
    uint16_t e_phnum;
    uint16_t e_shentsize;
    uint16_t e_shnum;
    uint16_t e_shstrndx;
};)

VPU_PACKED(mv_blob_header {
    uint32_t magic_number;
    uint32_t file_size;
    uint32_t blob_ver_major;
    uint32_t blob_ver_minor;
    uint32_t inputs_count;
    uint32_t outputs_count;
    uint32_t stages_count;
    uint32_t inputs_size;
    uint32_t outputs_size;
    uint32_t batch_size;
    uint32_t bss_mem_size;
    uint32_t number_of_cmx_slices;
    uint32_t number_of_shaves;
    uint32_t has_hw_stage;
    uint32_t has_shave_stage;
    uint32_t has_dma_stage;
    uint32_t input_info_section_offset;
    uint32_t output_info_section_offset;
    uint32_t stage_section_offset;
    uint32_t const_data_section_offset;
};)

VPU_PACKED(mv_stage_header {
    uint32_t stage_length;
    uint32_t stage_type;
    uint32_t numShaves;
};)

}  //  namespace vpu
