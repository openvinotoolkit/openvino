// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reader_writer.hpp"

INSTANTIATE_TEST_SUITE_P(WriterReaderUnitTests,
                         AllSections,
                         ::testing::Values(std::vector<uint16_t>{KnownSectionType::RUNTIME_REQUIREMENTS,
                                                                 KnownSectionType::BATCH_SIZE},
                                           std::vector<uint16_t>{KnownSectionType::RUNTIME_REQUIREMENTS,
                                                                 KnownSectionType::BATCH_SIZE,
                                                                 KnownSectionType::ELF_INIT_SCHEDULES}));

INSTANTIATE_TEST_SUITE_P(WriterReaderUnitTests,
                         IncompatibleCRE,
                         ::testing::Values(std::vector<uint16_t>(),
                                           std::vector<uint16_t>{KnownSectionType::RUNTIME_REQUIREMENTS},
                                           std::vector<uint16_t>{KnownSectionType::BATCH_SIZE}));
