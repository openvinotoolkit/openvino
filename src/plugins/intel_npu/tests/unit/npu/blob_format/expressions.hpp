// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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

const ov::Tensor DUMMY_MAIN_SCHEDULE(ov::element::u8, ov::Shape{16});
const auto ELF_MAIN_SCHEDULE_SECTION = std::make_shared<intel_npu::ELFMainScheduleSection>(DUMMY_MAIN_SCHEDULE);

std::vector<ov::Tensor> DUMMY_INIT_SCHEDULES = {ov::Tensor(ov::element::u8, ov::Shape{16})};
const auto ELF_INIT_SCHEDULES_SECTION = std::make_shared<intel_npu::ELFInitSchedulesSection>(DUMMY_INIT_SCHEDULES);

const std::vector<CRE::Token> expression_1 = {};

const std::vector<CRE::Token> expression_2 = {CRE::AND};

/*
        AND
         |
       *ELF*
*/
const std::vector<CRE::Token> expression_3 = {CRE::AND, CRE::ELF_SCHEDULE};

/*
           AND
          /   \
       *ELF*  *BT*
*/
const std::vector<CRE::Token> expression_4 = {CRE::AND, CRE::ELF_SCHEDULE, CRE::BATCHING};

/*
              AND
           /   |   \
        *ELF* *BT* *WS*
*/
const std::vector<CRE::Token> expression_5 = {CRE::AND, CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION};

/*
            AND
           /   \
        *ELF*  OR
              /  \
           *BT*  *WS*
*/
const std::vector<CRE::Token> expression_6 =
    {CRE::AND, CRE::ELF_SCHEDULE, CRE::OPEN, CRE::OR, CRE::BATCHING, CRE::WEIGHTS_SEPARATION, CRE::CLOSE};

/*
            OR
          /    \
        *ELF*  AND
              /   \
           *BT*   *WS*
*/
const std::vector<CRE::Token> expression_7 =
    {CRE::OR, CRE::ELF_SCHEDULE, CRE::OPEN, CRE::AND, CRE::BATCHING, CRE::WEIGHTS_SEPARATION, CRE::CLOSE};

/*
                ___ AND ___
               /     |      \
           *ELF*     OR      OR
                    /  \    /  \
                 *BT* *WS* *WS* *BT*
*/
const std::vector<CRE::Token> expression_8 = {CRE::AND,
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
        /    |    \          /   |    \
      *ELF* *WS* *BT*     *ELF* *WS*  AND
                                     /   \
                                   *ELF* *BT*
*/
const std::vector<CRE::Token> expression_9 = {CRE::OR,
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
                  ____ OR ____
                /             \
               /               \
              /                 \
         __ OR __             _ AND _
        /   |    \           /   |   \
      AND  *WS* *ELF*     *ELF* *WS* *BT*
     /   \
   *BT* *ELF*
*/
// expression_9 but with reversed leaves
const std::vector<CRE::Token> expression_10 = {CRE::OR,
                                               CRE::OPEN,
                                               CRE::OR,
                                               CRE::OPEN,
                                               CRE::AND,
                                               CRE::BATCHING,
                                               CRE::ELF_SCHEDULE,
                                               CRE::CLOSE,
                                               CRE::WEIGHTS_SEPARATION,
                                               CRE::ELF_SCHEDULE,
                                               CRE::CLOSE,
                                               CRE::OPEN,
                                               CRE::AND,
                                               CRE::ELF_SCHEDULE,
                                               CRE::WEIGHTS_SEPARATION,
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
const std::vector<CRE::Token> expression_11 =
    {CRE::AND, CRE::OPEN, CRE::AND, CRE::OPEN, CRE::OR, CRE::WEIGHTS_SEPARATION, CRE::CLOSE, CRE::CLOSE};

/*
             AND
            /   \
         *ELF*   OR
                /   \
             *WS*   OR
                   /   \
                  AND   *BT*
                 /  \
             *ELF* *WS*
*/
const std::vector<CRE::Token> expression_12 = {CRE::AND,
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

/*
              AND
           /   |   \
        *ELF* *BT* *ELF*
*/
// should we allow duplicated tokens?
const std::vector<CRE::Token> expression_13 = {CRE::AND, CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::ELF_SCHEDULE};

/*
    NOT
     |
   *ELF*
*/
const std::vector<CRE::Token> expression_14 = {CRE::NOT, CRE::ELF_SCHEDULE};

/*
        AND
         |
       ~ELF
*/
const std::vector<CRE::Token> expression_15 = {CRE::AND, CRE::NOT, CRE::ELF_SCHEDULE};

/*
              AND
           /   |   \
        ~ELF  ~BT  *WS*
*/
const std::vector<CRE::Token> expression_16 =
    {CRE::AND, CRE::NOT, CRE::ELF_SCHEDULE, CRE::NOT, CRE::BATCHING, CRE::WEIGHTS_SEPARATION};

/*
            AND
           /   \
        ~ELF  ~OR
              /  \
           *BT*  ~WS
*/
const std::vector<CRE::Token> expression_17 = {CRE::AND,
                                               CRE::NOT,
                                               CRE::ELF_SCHEDULE,
                                               CRE::NOT,
                                               CRE::OPEN,
                                               CRE::OR,
                                               CRE::BATCHING,
                                               CRE::NOT,
                                               CRE::WEIGHTS_SEPARATION,
                                               CRE::CLOSE};

/*
      NOT
       |
      AND
     /   \
  *ELF*  *BT*
*/
const std::vector<CRE::Token> expression_18 =
    {CRE::NOT, CRE::OPEN, CRE::AND, CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::CLOSE};

/*
                    NOT
                     |
                ___ AND ___
               /     |      \
           *ELF*     OR      OR
                    /  \    /  \
                 ~BT  *WS* *WS* ~BT
*/
const std::vector<CRE::Token> expression_19 = {CRE::NOT,
                                               CRE::OPEN,
                                               CRE::AND,
                                               CRE::ELF_SCHEDULE,
                                               CRE::OPEN,
                                               CRE::OR,
                                               CRE::NOT,
                                               CRE::BATCHING,
                                               CRE::WEIGHTS_SEPARATION,
                                               CRE::CLOSE,
                                               CRE::OPEN,
                                               CRE::OR,
                                               CRE::WEIGHTS_SEPARATION,
                                               CRE::NOT,
                                               CRE::BATCHING,
                                               CRE::CLOSE,
                                               CRE::CLOSE};

/*
                  _ OR _
                /        \
               /          \
             NOT           \
              |             \
              OR             OR
            /    \         /    \
         *ELF*  *BT*    *ELF*   *WS*
*/
const std::vector<CRE::Token> expression_20 = {CRE::OR,
                                               CRE::NOT,
                                               CRE::OPEN,
                                               CRE::OR,
                                               CRE::ELF_SCHEDULE,
                                               CRE::BATCHING,
                                               CRE::CLOSE,
                                               CRE::OPEN,
                                               CRE::OR,
                                               CRE::ELF_SCHEDULE,
                                               CRE::WEIGHTS_SEPARATION,
                                               CRE::CLOSE};

/*
                NOT
                 |
                NOT
                 |
                NOT
                 |
              _ AND _
             /       \
            /         \
           AND        OR
            |       /    \
          ~ELF    ~BT    ~WS
*/
const std::vector<CRE::Token> expression_21 = {CRE::NOT,
                                               CRE::NOT,
                                               CRE::NOT,
                                               CRE::OPEN,
                                               CRE::AND,
                                               CRE::OPEN,
                                               CRE::AND,
                                               CRE::NOT,
                                               CRE::ELF_SCHEDULE,
                                               CRE::CLOSE,
                                               CRE::OPEN,
                                               CRE::OR,
                                               CRE::NOT,
                                               CRE::BATCHING,
                                               CRE::NOT,
                                               CRE::WEIGHTS_SEPARATION,
                                               CRE::CLOSE,
                                               CRE::CLOSE};

// missing operand for OR operator
const std::vector<CRE::Token> invalid_expression_1 = {CRE::AND, CRE::ELF_SCHEDULE, CRE::OPEN, CRE::OR, CRE::CLOSE};

// missing closed parenthesis
const std::vector<CRE::Token> invalid_expression_2 =
    {CRE::AND, CRE::ELF_SCHEDULE, CRE::OPEN, CRE::OR, CRE::BATCHING, CRE::WEIGHTS_SEPARATION};

// missing open parenthesis
const std::vector<CRE::Token> invalid_expression_3 =
    {CRE::AND, CRE::ELF_SCHEDULE, CRE::OR, CRE::BATCHING, CRE::WEIGHTS_SEPARATION, CRE::CLOSE};

/*
                ___ AND ___
               /     |      \
           *ELF*     OR      OR
                     |      /  \
                     0    *WS* *BT*
*/
// missing operand for the first OR operator
const std::vector<CRE::Token> invalid_expression_4 = {CRE::AND,
                                                      CRE::ELF_SCHEDULE,
                                                      CRE::OPEN,
                                                      CRE::OR,
                                                      CRE::CLOSE,
                                                      CRE::OPEN,
                                                      CRE::OR,
                                                      CRE::WEIGHTS_SEPARATION,
                                                      CRE::BATCHING,
                                                      CRE::CLOSE};

// missing operands for nested operators
const std::vector<CRE::Token> invalid_expression_5 =
    {CRE::AND, CRE::OPEN, CRE::OR, CRE::OPEN, CRE::OR, CRE::CLOSE, CRE::CLOSE};

// NOT missing operand
const std::vector<CRE::Token> invalid_expression_6 = {CRE::AND, CRE::ELF_SCHEDULE, CRE::NOT};

// chained NOTs with no operand
const std::vector<CRE::Token> invalid_expression_7 = {CRE::NOT, CRE::NOT};

// NOT missing operand before CLOSE
const std::vector<CRE::Token> invalid_expression_8 = {CRE::AND, CRE::OPEN, CRE::NOT, CRE::CLOSE};
