// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cre.hpp"

#define MAKE_PARAM(expr, result, ...) \
    std::make_tuple(std::string(#expr), expr, std::vector<uint16_t>{__VA_ARGS__}, result)

const std::vector<CREToken> expression_1 = {};

const std::vector<CREToken> expression_3 = {SectionTypeCode::ELF_MAIN_SCHEDULE};

/*
           AND
          /   \
       *ELF*  *BT*
*/
const std::vector<CREToken> expression_4 = {SectionTypeCode::ELF_MAIN_SCHEDULE, CRE::AND, SectionTypeCode::BATCH_SIZE};

/*
              AND
           /   |   \
        *ELF* *BT* *WS*
*/
const std::vector<CREToken> expression_5 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                            CRE::AND,
                                            SectionTypeCode::BATCH_SIZE,
                                            CRE::AND,
                                            SectionTypeCode::ELF_INIT_SCHEDULES};

/*
            AND
           /   \
        *ELF*  OR
              /  \
           *BT*  *WS*
*/
const std::vector<CREToken> expression_6 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                            CRE::AND,
                                            CRE::OPEN,
                                            SectionTypeCode::BATCH_SIZE,
                                            CRE::OR,
                                            SectionTypeCode::ELF_INIT_SCHEDULES,
                                            CRE::CLOSE};

/*
            OR
          /    \
        *ELF*  AND
              /   \
           *BT*   *WS*
*/
const std::vector<CREToken> expression_7 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                            CRE::OR,
                                            CRE::OPEN,
                                            SectionTypeCode::BATCH_SIZE,
                                            CRE::AND,
                                            SectionTypeCode::ELF_INIT_SCHEDULES,
                                            CRE::CLOSE};

/*
                ___ AND ___
               /     |      \
           *ELF*     OR      OR
                    /  \    /  \
                 *BT* *WS* *WS* *BT*
*/
const std::vector<CREToken> expression_8 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                            CRE::AND,
                                            CRE::OPEN,
                                            SectionTypeCode::BATCH_SIZE,
                                            CRE::OR,
                                            SectionTypeCode::ELF_INIT_SCHEDULES,
                                            CRE::CLOSE,
                                            CRE::AND,
                                            CRE::OPEN,
                                            SectionTypeCode::ELF_INIT_SCHEDULES,
                                            CRE::OR,
                                            SectionTypeCode::BATCH_SIZE,
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
const std::vector<CREToken> expression_9 = {CRE::OPEN,
                                            SectionTypeCode::ELF_MAIN_SCHEDULE,
                                            CRE::AND,
                                            SectionTypeCode::ELF_INIT_SCHEDULES,
                                            CRE::AND,
                                            SectionTypeCode::BATCH_SIZE,
                                            CRE::CLOSE,
                                            CRE::OR,
                                            CRE::OPEN,
                                            SectionTypeCode::ELF_MAIN_SCHEDULE,
                                            CRE::OR,
                                            SectionTypeCode::ELF_INIT_SCHEDULES,
                                            CRE::OR,
                                            CRE::OPEN,
                                            SectionTypeCode::ELF_MAIN_SCHEDULE,
                                            CRE::AND,
                                            SectionTypeCode::BATCH_SIZE,
                                            CRE::CLOSE,
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
const std::vector<CREToken> expression_10 = {CRE::OPEN,
                                             CRE::OPEN,
                                             SectionTypeCode::BATCH_SIZE,
                                             CRE::AND,
                                             SectionTypeCode::ELF_MAIN_SCHEDULE,
                                             CRE::CLOSE,
                                             CRE::OR,
                                             SectionTypeCode::ELF_INIT_SCHEDULES,
                                             CRE::OR,
                                             SectionTypeCode::ELF_MAIN_SCHEDULE,
                                             CRE::CLOSE,
                                             CRE::OR,
                                             CRE::OPEN,
                                             SectionTypeCode::ELF_MAIN_SCHEDULE,
                                             CRE::AND,
                                             SectionTypeCode::ELF_INIT_SCHEDULES,
                                             CRE::AND,
                                             SectionTypeCode::BATCH_SIZE,
                                             CRE::CLOSE};

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
const std::vector<CREToken> expression_12 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                             CRE::AND,
                                             CRE::OPEN,
                                             SectionTypeCode::ELF_INIT_SCHEDULES,
                                             CRE::OR,
                                             CRE::OPEN,
                                             CRE::OPEN,
                                             SectionTypeCode::ELF_MAIN_SCHEDULE,
                                             CRE::AND,
                                             SectionTypeCode::ELF_INIT_SCHEDULES,
                                             CRE::CLOSE,
                                             CRE::OR,
                                             SectionTypeCode::BATCH_SIZE,
                                             CRE::CLOSE,
                                             CRE::CLOSE};

/*
              AND
           /   |   \
        *ELF* *BT* *ELF*
*/
const std::vector<CREToken> expression_13 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                             CRE::AND,
                                             SectionTypeCode::BATCH_SIZE,
                                             CRE::AND,
                                             SectionTypeCode::ELF_MAIN_SCHEDULE};

/*
    NOT
     |
   *ELF*
*/
const std::vector<CREToken> expression_14 = {CRE::NOT, SectionTypeCode::ELF_MAIN_SCHEDULE};

/*
              AND
           /   |   \
        ~ELF  ~BT  *WS*
*/
const std::vector<CREToken> expression_16 = {CRE::NOT,
                                             SectionTypeCode::ELF_MAIN_SCHEDULE,
                                             CRE::AND,
                                             CRE::NOT,
                                             SectionTypeCode::BATCH_SIZE,
                                             CRE::AND,
                                             SectionTypeCode::ELF_INIT_SCHEDULES};

/*
            AND
           /   \
        ~ELF  ~OR
              /  \
           *BT*  ~WS
*/
const std::vector<CREToken> expression_17 = {CRE::NOT,
                                             SectionTypeCode::ELF_MAIN_SCHEDULE,
                                             CRE::AND,
                                             CRE::NOT,
                                             CRE::OPEN,
                                             SectionTypeCode::BATCH_SIZE,
                                             CRE::OR,
                                             CRE::NOT,
                                             SectionTypeCode::ELF_INIT_SCHEDULES,
                                             CRE::CLOSE};

/*
      NOT
       |
      AND
     /   \
  *ELF*  *BT*
*/
const std::vector<CREToken> expression_18 =
    {CRE::NOT, CRE::OPEN, SectionTypeCode::ELF_MAIN_SCHEDULE, CRE::AND, SectionTypeCode::BATCH_SIZE, CRE::CLOSE};

/*
    AND
    /  \
~ELF  *BT*
*/
const std::vector<CREToken> expression_15 = {CRE::NOT,
                                             SectionTypeCode::ELF_MAIN_SCHEDULE,
                                             CRE::AND,
                                             SectionTypeCode::BATCH_SIZE};

/*
                    NOT
                     |
                ___ AND ___
               /     |      \
           *ELF*     OR      OR
                    /  \    /  \
                 ~BT  *WS* *WS* ~BT
*/
const std::vector<CREToken> expression_19 = {CRE::NOT,
                                             CRE::OPEN,
                                             SectionTypeCode::ELF_MAIN_SCHEDULE,
                                             CRE::AND,
                                             CRE::OPEN,
                                             CRE::NOT,
                                             SectionTypeCode::BATCH_SIZE,
                                             CRE::OR,
                                             SectionTypeCode::ELF_INIT_SCHEDULES,
                                             CRE::CLOSE,
                                             CRE::AND,
                                             CRE::OPEN,
                                             SectionTypeCode::ELF_INIT_SCHEDULES,
                                             CRE::OR,
                                             CRE::NOT,
                                             SectionTypeCode::BATCH_SIZE,
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
const std::vector<CREToken> expression_20 = {CRE::NOT,
                                             CRE::OPEN,
                                             SectionTypeCode::ELF_MAIN_SCHEDULE,
                                             CRE::OR,
                                             SectionTypeCode::BATCH_SIZE,
                                             CRE::CLOSE,
                                             CRE::OR,
                                             CRE::OPEN,
                                             SectionTypeCode::ELF_MAIN_SCHEDULE,
                                             CRE::OR,
                                             SectionTypeCode::ELF_INIT_SCHEDULES,
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
          ~ELF        OR
                    /    \
                  ~BT    ~WS
*/
const std::vector<CREToken> expression_21 = {CRE::NOT,
                                             CRE::NOT,
                                             CRE::NOT,
                                             CRE::OPEN,
                                             CRE::NOT,
                                             SectionTypeCode::ELF_MAIN_SCHEDULE,
                                             CRE::AND,
                                             CRE::OPEN,
                                             CRE::NOT,
                                             SectionTypeCode::BATCH_SIZE,
                                             CRE::OR,
                                             CRE::NOT,
                                             SectionTypeCode::ELF_INIT_SCHEDULES,
                                             CRE::CLOSE,
                                             CRE::CLOSE};

const std::vector<CREToken> expression_22 = {CRE::OPEN, SectionTypeCode::ELF_MAIN_SCHEDULE, CRE::CLOSE};

const std::vector<CREToken> expression_23 =
    {CRE::OPEN, CRE::OPEN, CRE::NOT, SectionTypeCode::ELF_MAIN_SCHEDULE, CRE::CLOSE, CRE::CLOSE};

// missing both operands for the OR operator
const std::vector<CREToken> invalid_expression_1 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                    CRE::AND,
                                                    CRE::OPEN,
                                                    CRE::OR,
                                                    CRE::CLOSE};

// Missing only the first operand for the OR operator
const std::vector<CREToken> invalid_expression_15 =
    {SectionTypeCode::ELF_MAIN_SCHEDULE, CRE::AND, CRE::OPEN, SectionTypeCode::ELF_MAIN_SCHEDULE, CRE::OR, CRE::CLOSE};

// Missing only the second operand for the OR operator
const std::vector<CREToken> invalid_expression_16 =
    {SectionTypeCode::ELF_MAIN_SCHEDULE, CRE::AND, CRE::OPEN, CRE::OR, SectionTypeCode::ELF_MAIN_SCHEDULE, CRE::CLOSE};

// missing closed parenthesis
const std::vector<CREToken> invalid_expression_2 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                    CRE::AND,
                                                    CRE::OPEN,
                                                    SectionTypeCode::BATCH_SIZE,
                                                    CRE::OR,
                                                    SectionTypeCode::ELF_INIT_SCHEDULES};

// missing open parenthesis
const std::vector<CREToken> invalid_expression_3 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                    CRE::AND,
                                                    SectionTypeCode::BATCH_SIZE,
                                                    CRE::OR,
                                                    SectionTypeCode::ELF_INIT_SCHEDULES,
                                                    CRE::CLOSE};

/*
                ___ AND ___
               /     |      \
           *ELF*     OR      OR
                     |      /  \
                     0    *WS* *BT*
*/
// missing operand for the first OR operator
const std::vector<CREToken> invalid_expression_4 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                    CRE::AND,
                                                    CRE::OPEN,
                                                    CRE::OR,
                                                    CRE::CLOSE,
                                                    CRE::OPEN,
                                                    SectionTypeCode::ELF_INIT_SCHEDULES,
                                                    CRE::OR,
                                                    SectionTypeCode::BATCH_SIZE,
                                                    CRE::CLOSE};

// missing operands for nested operators
const std::vector<CREToken> invalid_expression_5 =
    {CRE::OPEN, CRE::OR, CRE::OPEN, CRE::OR, CRE::CLOSE, CRE::CLOSE, CRE::AND};

// NOT missing operand
const std::vector<CREToken> invalid_expression_6 = {SectionTypeCode::ELF_MAIN_SCHEDULE, CRE::AND, CRE::NOT};

// chained NOTs with no operand
const std::vector<CREToken> invalid_expression_7 = {CRE::NOT, CRE::NOT};

// NOT missing operand before CLOSE
const std::vector<CREToken> invalid_expression_8 = {CRE::OPEN, CRE::NOT, CRE::CLOSE};

// missing operand
const std::vector<CREToken> invalid_expression_9 = {CRE::AND};

// too many operands
const std::vector<CREToken> invalid_expression_10 = {CRE::NOT,
                                                     SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                     SectionTypeCode::BATCH_SIZE};

// missing CLOSE
const std::vector<CREToken> invalid_expression_11 = {CRE::OPEN,
                                                     CRE::OPEN,
                                                     CRE::NOT,
                                                     SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                     CRE::CLOSE};

// missing OPEN
const std::vector<CREToken> invalid_expression_12 = {CRE::OPEN,
                                                     CRE::NOT,
                                                     SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                     CRE::CLOSE,
                                                     CRE::CLOSE};

// Empty parrentheses cannot play the role of an operand
const std::vector<CREToken> invalid_expression_13 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                     CRE::AND,
                                                     CRE::OPEN,
                                                     CRE::CLOSE};

// The subexpression is just "NOT". The operand is missing
const std::vector<CREToken> invalid_expression_14 = {CRE::OPEN,
                                                     CRE::NOT,
                                                     CRE::CLOSE,
                                                     SectionTypeCode::ELF_MAIN_SCHEDULE};

// AND has too many operands
const std::vector<CREToken> invalid_expression_17 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                     CRE::AND,
                                                     SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                     SectionTypeCode::ELF_MAIN_SCHEDULE};

// OR has too many operands
const std::vector<CREToken> invalid_expression_18 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                     CRE::OR,
                                                     SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                     SectionTypeCode::ELF_MAIN_SCHEDULE};

// No operator to tie the two tokens
const std::vector<CREToken> invalid_expression_19 = {SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                     SectionTypeCode::ELF_MAIN_SCHEDULE};

// No operator to tie the two subexpressions
const std::vector<CREToken> invalid_expression_20 = {CRE::OPEN,
                                                     SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                     CRE::CLOSE,
                                                     CRE::OPEN,
                                                     SectionTypeCode::ELF_MAIN_SCHEDULE,
                                                     CRE::CLOSE};

// "OR" cannot replace an operand
const std::vector<CREToken> invalid_expression_21 = {SectionTypeCode::ELF_MAIN_SCHEDULE, CRE::OR, CRE::OR};

std::vector<CREParams> valid_test_cases = {
    MAKE_PARAM(expression_1, true),

    MAKE_PARAM(expression_3, true, SectionTypeCode::ELF_MAIN_SCHEDULE),
    MAKE_PARAM(expression_3, false),

    MAKE_PARAM(expression_4, true, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_4, false, SectionTypeCode::ELF_MAIN_SCHEDULE),
    MAKE_PARAM(expression_4, false, SectionTypeCode::BATCH_SIZE),

    MAKE_PARAM(expression_5,
               true,
               SectionTypeCode::ELF_MAIN_SCHEDULE,
               SectionTypeCode::BATCH_SIZE,
               SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_5, false, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_5, false, SectionTypeCode::ELF_MAIN_SCHEDULE),

    MAKE_PARAM(expression_6,
               true,
               SectionTypeCode::ELF_MAIN_SCHEDULE,
               SectionTypeCode::BATCH_SIZE,
               SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_6, true, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_6, true, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_6, false, SectionTypeCode::BATCH_SIZE, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_6, false),

    MAKE_PARAM(expression_7,
               true,
               SectionTypeCode::ELF_MAIN_SCHEDULE,
               SectionTypeCode::BATCH_SIZE,
               SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_7, true, SectionTypeCode::BATCH_SIZE, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_7, true, SectionTypeCode::ELF_MAIN_SCHEDULE),
    MAKE_PARAM(expression_7, false, SectionTypeCode::ELF_INIT_SCHEDULES),

    MAKE_PARAM(expression_8,
               true,
               SectionTypeCode::ELF_MAIN_SCHEDULE,
               SectionTypeCode::BATCH_SIZE,
               SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_8, true, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_8, true, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_8, false, SectionTypeCode::BATCH_SIZE, SectionTypeCode::ELF_INIT_SCHEDULES),

    MAKE_PARAM(expression_9,
               true,
               SectionTypeCode::ELF_MAIN_SCHEDULE,
               SectionTypeCode::BATCH_SIZE,
               SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_9, true, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_9, true, SectionTypeCode::ELF_MAIN_SCHEDULE),
    MAKE_PARAM(expression_9, false, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_9, false),

    // should have the same behavior as expression_9
    MAKE_PARAM(expression_10,
               true,
               SectionTypeCode::ELF_MAIN_SCHEDULE,
               SectionTypeCode::BATCH_SIZE,
               SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_10, true, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_10, true, SectionTypeCode::ELF_MAIN_SCHEDULE),
    MAKE_PARAM(expression_10, false, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_10, false),

    MAKE_PARAM(expression_12,
               true,
               SectionTypeCode::ELF_MAIN_SCHEDULE,
               SectionTypeCode::BATCH_SIZE,
               SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_12, true, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_12, true, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_12, false, SectionTypeCode::BATCH_SIZE, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_12, false, SectionTypeCode::ELF_MAIN_SCHEDULE),

    MAKE_PARAM(expression_13, true, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::BATCH_SIZE),

    MAKE_PARAM(expression_14, true, SectionTypeCode::ELF_INIT_SCHEDULES, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_14, true),
    MAKE_PARAM(expression_14, false, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_14, false, SectionTypeCode::ELF_MAIN_SCHEDULE),

    MAKE_PARAM(expression_15, false, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_15, false, SectionTypeCode::ELF_MAIN_SCHEDULE),
    MAKE_PARAM(expression_15, true, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_15, false),

    MAKE_PARAM(expression_16, true, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_16,
               false,
               SectionTypeCode::ELF_MAIN_SCHEDULE,
               SectionTypeCode::BATCH_SIZE,
               SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_16, false, SectionTypeCode::BATCH_SIZE, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_16, false),

    MAKE_PARAM(expression_17, true, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_17,
               false,
               SectionTypeCode::ELF_MAIN_SCHEDULE,
               SectionTypeCode::BATCH_SIZE,
               SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_17, false, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_17, false, SectionTypeCode::BATCH_SIZE, SectionTypeCode::ELF_INIT_SCHEDULES),

    MAKE_PARAM(expression_18, false, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_18, true, SectionTypeCode::ELF_MAIN_SCHEDULE),
    MAKE_PARAM(expression_18, true),

    MAKE_PARAM(expression_19, true, SectionTypeCode::BATCH_SIZE, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_19, true, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_19, true, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_19, true, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_19, true),
    MAKE_PARAM(expression_19, false, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::ELF_INIT_SCHEDULES),

    MAKE_PARAM(expression_20,
               true,
               SectionTypeCode::ELF_MAIN_SCHEDULE,
               SectionTypeCode::BATCH_SIZE,
               SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_20, true, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_20, true),
    MAKE_PARAM(expression_20, false, SectionTypeCode::BATCH_SIZE),

    MAKE_PARAM(expression_21,
               true,
               SectionTypeCode::ELF_MAIN_SCHEDULE,
               SectionTypeCode::BATCH_SIZE,
               SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_21, true, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_21, true, SectionTypeCode::BATCH_SIZE, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_21, true, SectionTypeCode::ELF_MAIN_SCHEDULE, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_21, true, SectionTypeCode::ELF_MAIN_SCHEDULE),
    MAKE_PARAM(expression_21, false, SectionTypeCode::ELF_INIT_SCHEDULES),
    MAKE_PARAM(expression_21, false, SectionTypeCode::BATCH_SIZE),
    MAKE_PARAM(expression_21, false),

    MAKE_PARAM(expression_22, true, SectionTypeCode::ELF_MAIN_SCHEDULE),
    MAKE_PARAM(expression_23, false, SectionTypeCode::ELF_MAIN_SCHEDULE),
};

INSTANTIATE_TEST_SUITE_P(CRE, ValidExpression, ::testing::ValuesIn(valid_test_cases), CREUnitTests::getTestCaseName);

std::vector<CREParams> invalid_test_cases = {
    MAKE_PARAM(invalid_expression_1, false),  MAKE_PARAM(invalid_expression_2, false),
    MAKE_PARAM(invalid_expression_3, false),  MAKE_PARAM(invalid_expression_4, false),
    MAKE_PARAM(invalid_expression_5, false),  MAKE_PARAM(invalid_expression_6, false),
    MAKE_PARAM(invalid_expression_7, false),  MAKE_PARAM(invalid_expression_8, false),
    MAKE_PARAM(invalid_expression_9, false),  MAKE_PARAM(invalid_expression_10, false),
    MAKE_PARAM(invalid_expression_11, false), MAKE_PARAM(invalid_expression_12, false),
    MAKE_PARAM(invalid_expression_13, false), MAKE_PARAM(invalid_expression_14, false),
    MAKE_PARAM(invalid_expression_15, false), MAKE_PARAM(invalid_expression_16, false),
    MAKE_PARAM(invalid_expression_17, false), MAKE_PARAM(invalid_expression_18, false),
    MAKE_PARAM(invalid_expression_19, false), MAKE_PARAM(invalid_expression_20, false),
    MAKE_PARAM(invalid_expression_21, false),
};

INSTANTIATE_TEST_SUITE_P(CRE,
                         InvalidExpression,
                         ::testing::ValuesIn(invalid_test_cases),
                         CREUnitTests::getTestCaseName);
