// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reader_writer.hpp"

INSTANTIATE_TEST_SUITE_P(WriterReaderUnitTests,
                         AllSections,
                         ::testing::Values(std::vector<uint16_t>{PredefinedSectionType::CRE,
                                                                 PredefinedSectionType::BATCH_SIZE},
                                           std::vector<uint16_t>{PredefinedSectionType::CRE,
                                                                 PredefinedSectionType::BATCH_SIZE,
                                                                 PredefinedSectionType::ELF_INIT_SCHEDULES}));

INSTANTIATE_TEST_SUITE_P(WriterReaderUnitTests,
                         IncompatibleCRE,
                         ::testing::Values(std::vector<uint16_t>(),
                                           std::vector<uint16_t>{PredefinedSectionType::CRE},
                                           std::vector<uint16_t>{PredefinedSectionType::BATCH_SIZE}));
