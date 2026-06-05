// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reader_writer.hpp"

INSTANTIATE_TEST_SUITE_P(
    WriterReaderUnitTests,
    AllSections,
    ::testing::Values(make_tuple((std::vector<CRE::Token>{}), std::vector<uint16_t>{PredefinedSectionType::CRE}),
                      make_tuple(std::vector<CRE::Token>{CRE::AND, PredefinedSectionType::BATCH_SIZE},
                                 std::vector<uint16_t>{PredefinedSectionType::CRE, PredefinedSectionType::BATCH_SIZE}),
                      make_tuple(std::vector<CRE::Token>{CRE::AND, PredefinedSectionType::ELF_INIT_SCHEDULES},
                                 std::vector<uint16_t>{PredefinedSectionType::CRE,
                                                       PredefinedSectionType::ELF_INIT_SCHEDULES}),
                      make_tuple(std::vector<CRE::Token>{CRE::AND,
                                                         PredefinedSectionType::BATCH_SIZE,
                                                         PredefinedSectionType::ELF_INIT_SCHEDULES},
                                 std::vector<uint16_t>{PredefinedSectionType::CRE,
                                                       PredefinedSectionType::BATCH_SIZE,
                                                       PredefinedSectionType::ELF_INIT_SCHEDULES})));

INSTANTIATE_TEST_SUITE_P(
    WriterReaderUnitTests,
    IncompatibleCRE,
    ::testing::Values(make_tuple(std::vector<CRE::Token>{CRE::AND, PredefinedSectionType::BATCH_SIZE},
                                 std::vector<uint16_t>()),
                      make_tuple(std::vector<CRE::Token>{CRE::AND, PredefinedSectionType::ELF_MAIN_SCHEDULE},
                                 std::vector<uint16_t>{PredefinedSectionType::BATCH_SIZE}),
                      make_tuple(std::vector<CRE::Token>{CRE::AND,
                                                         PredefinedSectionType::BATCH_SIZE,
                                                         PredefinedSectionType::ELF_INIT_SCHEDULES},
                                 std::vector<uint16_t>{PredefinedSectionType::BATCH_SIZE})));
