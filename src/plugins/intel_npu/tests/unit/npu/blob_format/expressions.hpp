// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_size_section.hpp"
#include "compiler_schedules_sections.hpp"
#include "intel_npu/common/cre.hpp"
#include "intel_npu/common/offsets_table.hpp"
#include "io_layouts_section.hpp"

using namespace intel_npu;

const std::optional<int64_t> BATCH = 2;

const auto BATCH_SECTION = std::make_shared<intel_npu::BatchSizeSection>(BATCH.value());

const std::vector<CRE::Token> DUMMY_CRE_EXPRESSION = {CRE::OPEN, CRE::AND, CRE::ELF_SCHEDULE, CRE::CLOSE};
const auto CRE_SECTION = std::make_shared<intel_npu::CRESection>(CRE(DUMMY_CRE_EXPRESSION));

const intel_npu::OffsetsTable DUMMY_OFFSETS_TABLE;
const auto OFFSETS_TABLE_SECTION = std::make_shared<intel_npu::OffsetsTableSection>(DUMMY_OFFSETS_TABLE);

const std::vector<ov::Layout> DUMMY_INPUT_LAYOUTS = {ov::Layout("NCHW")};
const std::vector<ov::Layout> DUMMY_OUTPUT_LAYOUTS = {ov::Layout("NCHW")};
const auto IO_LAYOUTS_SECTION =
    std::make_shared<intel_npu::IOLayoutsSection>(DUMMY_INPUT_LAYOUTS, DUMMY_OUTPUT_LAYOUTS);

const ov::Tensor DUMMY_MAIN_SCHEDULE(ov::element::u8, ov::Shape{16});
const auto ELF_MAIN_SCHEDULE_SECTION = std::make_shared<intel_npu::ELFMainScheduleSection>(DUMMY_MAIN_SCHEDULE);

std::vector<ov::Tensor> DUMMY_INIT_SCHEDULES = {ov::Tensor(ov::element::u8, ov::Shape{16})};
const auto ELF_INIT_SCHEDULES_SECTION = std::make_shared<intel_npu::ELFInitSchedulesSection>(DUMMY_INIT_SCHEDULES);

/*
        AND
         |
       *ELF*
*/
const std::vector<CRE::Token> expression_1 = {CRE::AND, CRE::ELF_SCHEDULE};

/*
           AND
          /   \
       *ELF*  *BS*
*/
const std::vector<CRE::Token> expression_2 = {CRE::AND, CRE::ELF_SCHEDULE, CRE::BATCHING};

/*
              AND
           /   |   \
        *ELF* *BS* *WS*
*/
const std::vector<CRE::Token> expression_3 = {CRE::AND, CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION};

/*
            AND
           /   \
        *ELF*  OR
              /  \
           *BS*  *WS*
*/
const std::vector<CRE::Token> expression_4 =
    {CRE::AND, CRE::ELF_SCHEDULE, CRE::OPEN, CRE::OR, CRE::BATCHING, CRE::WEIGHTS_SEPARATION, CRE::CLOSE};

/*
            OR
          /    \
        *ELF*  AND
              /   \
           *BS*   *WS*
*/
const std::vector<CRE::Token> expression_5 =
    {CRE::OR, CRE::ELF_SCHEDULE, CRE::OPEN, CRE::AND, CRE::BATCHING, CRE::WEIGHTS_SEPARATION, CRE::CLOSE};

/*
                ___ AND ___
               /     |      \
           *ELF*     OR      OR
                    /  \    /  \
                 *BS* *WS* *WS* *BS*
*/
const std::vector<CRE::Token> expression_6 = {CRE::AND,
                                              CRE::ELF_SCHEDULE,
                                              CRE::OPEN,
                                              CRE::OR,
                                              CRE::BATCHING,
                                              CRE::WEIGHTS_SEPARATION,
                                              CRE::CLOSE,
                                              CRE::OPEN,
                                              CRE::OR,
                                              CRE::WEIGHTS_SEPARATION,
                                              CRE::BATCHING,
                                              CRE::CLOSE};

/*
                  ____ OR ____
                /             \
               /               \
              /                 \
         __ AND __             _ OR _
        /    |    \           /   |   \
      *ELF* *WS* *BS*      *ELF* *WS*  AND
                                      /   \
                                    *ELF* *BS*
*/
const std::vector<CRE::Token> expression_7 = {CRE::OR,
                                              CRE::OPEN,
                                              CRE::AND,
                                              CRE::ELF_SCHEDULE,
                                              CRE::WEIGHTS_SEPARATION,
                                              CRE::BATCHING,
                                              CRE::CLOSE,
                                              CRE::OPEN,
                                              CRE::OR,
                                              CRE::ELF_SCHEDULE,
                                              CRE::WEIGHTS_SEPARATION,
                                              CRE::CLOSE,
                                              CRE::OPEN,
                                              CRE::AND,
                                              CRE::ELF_SCHEDULE,
                                              CRE::BATCHING,
                                              CRE::CLOSE};

/*
    AND
     |
    AND
     |
    OR
     |
   *WS*
*/
const std::vector<CRE::Token> expression_8 =
    {CRE::AND, CRE::OPEN, CRE::AND, CRE::OPEN, CRE::OR, CRE::WEIGHTS_SEPARATION, CRE::CLOSE, CRE::CLOSE};

/*
             AND
            /   \
         *ELF*   OR
                /   \
             *WS*   OR
                   /   \
                  AND   *BS*
                 /  \
             *ELF* *WS*
*/
const std::vector<CRE::Token> expression_9 = {CRE::AND,
                                              CRE::ELF_SCHEDULE,
                                              CRE::OPEN,
                                              CRE::OR,
                                              CRE::WEIGHTS_SEPARATION,
                                              CRE::OPEN,
                                              CRE::OR,
                                              CRE::OPEN,
                                              CRE::AND,
                                              CRE::ELF_SCHEDULE,
                                              CRE::WEIGHTS_SEPARATION,
                                              CRE::CLOSE,
                                              CRE::BATCHING,
                                              CRE::CLOSE,
                                              CRE::CLOSE};
