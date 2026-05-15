// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reader_writer.hpp"

INSTANTIATE_TEST_SUITE_P(
    WriterReaderUnitTests,
    AllSections,
    ::testing::Values(make_tuple((std::vector<CRE::Token>{}), std::vector<uint16_t>{CRE::CRE_EVALUATION}),
                      make_tuple(std::vector<CRE::Token>{CRE::AND, CRE::BATCHING},
                                 std::vector<uint16_t>{CRE::CRE_EVALUATION, CRE::BATCHING}),
                      make_tuple(std::vector<CRE::Token>{CRE::AND, CRE::WEIGHTS_SEPARATION},
                                 std::vector<uint16_t>{CRE::CRE_EVALUATION, CRE::WEIGHTS_SEPARATION}),
                      make_tuple(std::vector<CRE::Token>{CRE::AND, CRE::BATCHING, CRE::WEIGHTS_SEPARATION},
                                 std::vector<uint16_t>{CRE::CRE_EVALUATION, CRE::BATCHING, CRE::WEIGHTS_SEPARATION})));

INSTANTIATE_TEST_SUITE_P(
    WriterReaderUnitTests,
    IncompatibleCRE,
    ::testing::Values(make_tuple(std::vector<CRE::Token>{CRE::AND, CRE::BATCHING}, std::vector<uint16_t>()),
                      make_tuple(std::vector<CRE::Token>{CRE::AND, CRE::ELF_SCHEDULE},
                                 std::vector<uint16_t>{CRE::BATCHING}),
                      make_tuple(std::vector<CRE::Token>{CRE::AND, CRE::BATCHING, CRE::WEIGHTS_SEPARATION},
                                 std::vector<uint16_t>{CRE::BATCHING})));
