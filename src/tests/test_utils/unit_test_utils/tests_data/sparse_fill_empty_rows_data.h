#pragma once

#define LIST(...) \
    { __VA_ARGS__ }

//TEST_DATA(indicesData,
//          valuesData,
//          denseShapeData,
//          expectedIndicesOutput,
//          expectedValuesOutput,
//          expectedEmptyRowIndicatorOutput,
//          testcaseName)

// NOTE: expected output were generated using TensorFlow.

TEST_DATA(LIST(0, 1, 0, 0, 1, 1),
          LIST(1, 2, 3),
          LIST(2, 5),
          LIST(0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1),
          LIST(1, 2, 42, 42, 3, 42),
          LIST(0, 1),
          "DebugTest")