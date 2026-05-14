// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cre.hpp"

#define MAKE_PARAM(expr, result, ...) \
    std::make_tuple(std::string(#expr), expr, std::vector<uint16_t>{__VA_ARGS__}, result)

const std::vector<CRE::Token> expression_1 = {};

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

// missing operand
const std::vector<CRE::Token> invalid_expression_9 = {CRE::AND};

std::vector<CREParams> valid_test_cases = {
    MAKE_PARAM(expression_1, true),

    MAKE_PARAM(expression_3, true, CRE::ELF_SCHEDULE),
    MAKE_PARAM(expression_3, false),

    MAKE_PARAM(expression_4, true, CRE::ELF_SCHEDULE, CRE::BATCHING),
    MAKE_PARAM(expression_4, false, CRE::ELF_SCHEDULE),
    MAKE_PARAM(expression_4, false, CRE::BATCHING),

    MAKE_PARAM(expression_5, true, CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_5, false, CRE::ELF_SCHEDULE, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_5, false, CRE::ELF_SCHEDULE),

    MAKE_PARAM(expression_6, true, CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_6, true, CRE::ELF_SCHEDULE, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_6, true, CRE::ELF_SCHEDULE, CRE::BATCHING),
    MAKE_PARAM(expression_6, false, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_6, false),

    MAKE_PARAM(expression_7, true, CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_7, true, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_7, true, CRE::ELF_SCHEDULE),
    MAKE_PARAM(expression_7, false, CRE::WEIGHTS_SEPARATION),

    MAKE_PARAM(expression_8, true, CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_8, true, CRE::ELF_SCHEDULE, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_8, true, CRE::ELF_SCHEDULE, CRE::BATCHING),
    MAKE_PARAM(expression_8, false, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),

    MAKE_PARAM(expression_9, true, CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_9, true, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_9, true, CRE::ELF_SCHEDULE),
    MAKE_PARAM(expression_9, false, CRE::BATCHING),
    MAKE_PARAM(expression_9, false),

    // should have the same behavior as expression_9
    MAKE_PARAM(expression_10, true, CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_10, true, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_10, true, CRE::ELF_SCHEDULE),
    MAKE_PARAM(expression_10, false, CRE::BATCHING),
    MAKE_PARAM(expression_10, false),

    MAKE_PARAM(expression_11, true, CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_11, true, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_11, false),

    MAKE_PARAM(expression_12, true, CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_12, true, CRE::ELF_SCHEDULE, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_12, true, CRE::ELF_SCHEDULE, CRE::BATCHING),
    MAKE_PARAM(expression_12, false, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_12, false, CRE::ELF_SCHEDULE),

    MAKE_PARAM(expression_13, true, CRE::ELF_SCHEDULE, CRE::BATCHING),

    MAKE_PARAM(expression_14, true, CRE::WEIGHTS_SEPARATION, CRE::BATCHING),
    MAKE_PARAM(expression_14, true),
    MAKE_PARAM(expression_14, false, CRE::ELF_SCHEDULE, CRE::BATCHING),
    MAKE_PARAM(expression_14, false, CRE::ELF_SCHEDULE),

    MAKE_PARAM(expression_15, true, CRE::WEIGHTS_SEPARATION, CRE::BATCHING),
    MAKE_PARAM(expression_15, true),
    MAKE_PARAM(expression_15, false, CRE::ELF_SCHEDULE),

    MAKE_PARAM(expression_16, true, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_16, false, CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_16, false, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_16, false),

    MAKE_PARAM(expression_17, true, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_17, false, CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_17, false, CRE::ELF_SCHEDULE, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_17, false, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),

    MAKE_PARAM(expression_18, false, CRE::ELF_SCHEDULE, CRE::BATCHING),
    MAKE_PARAM(expression_18, true, CRE::ELF_SCHEDULE),
    MAKE_PARAM(expression_18, true),

    MAKE_PARAM(expression_19, true, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_19, true, CRE::ELF_SCHEDULE, CRE::BATCHING),
    MAKE_PARAM(expression_19, true, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_19, true, CRE::BATCHING),
    MAKE_PARAM(expression_19, true),
    MAKE_PARAM(expression_19, false, CRE::ELF_SCHEDULE, CRE::WEIGHTS_SEPARATION),

    MAKE_PARAM(expression_20, true, CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_20, true, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_20, true),
    MAKE_PARAM(expression_20, false, CRE::BATCHING),

    MAKE_PARAM(expression_21, true, CRE::ELF_SCHEDULE, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_21, true, CRE::ELF_SCHEDULE, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_21, true, CRE::BATCHING, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_21, true, CRE::ELF_SCHEDULE, CRE::BATCHING),
    MAKE_PARAM(expression_21, true, CRE::ELF_SCHEDULE),
    MAKE_PARAM(expression_21, false, CRE::WEIGHTS_SEPARATION),
    MAKE_PARAM(expression_21, false, CRE::BATCHING),
    MAKE_PARAM(expression_21, false),
};

INSTANTIATE_TEST_SUITE_P(CRE, ValidExpression, ::testing::ValuesIn(valid_test_cases), CREUnitTests::getTestCaseName);

std::vector<CREParams> invalid_test_cases = {
    MAKE_PARAM(invalid_expression_1, false),
    MAKE_PARAM(invalid_expression_2, false),
    MAKE_PARAM(invalid_expression_3, false),
    MAKE_PARAM(invalid_expression_4, false),
    MAKE_PARAM(invalid_expression_5, false),
    MAKE_PARAM(invalid_expression_6, false),
    MAKE_PARAM(invalid_expression_7, false),
    MAKE_PARAM(invalid_expression_8, false),
    MAKE_PARAM(invalid_expression_9, false),
};

INSTANTIATE_TEST_SUITE_P(CRE,
                         InvalidExpression,
                         ::testing::ValuesIn(invalid_test_cases),
                         CREUnitTests::getTestCaseName);
