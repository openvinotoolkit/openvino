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

// Expected outputs were generated using TensorFlow

TEST_DATA(LIST(0, 1, 0, 3, 2, 2),
          LIST(1, 2, 3),
          LIST(3, 5),
          LIST(0, 1, 0, 3, 1, 0, 2, 2),
          LIST(1, 2, 42, 3),
          LIST(0, 1, 0),
          "DebugTest")