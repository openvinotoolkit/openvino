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
          LIST(1.0, 2.0, 42.0, 3.0),
          LIST(0, 1, 0),
          "BasicCase");

TEST_DATA(LIST(0, 0, 1, 1, 2, 2),
          LIST(10, 20, 30),
          LIST(3, 3),
          LIST(0, 0, 1, 1, 2, 2),
          LIST(10.0, 20.0, 30.0),
          LIST(0, 0, 0),
          "NoEmptyRows");

TEST_DATA(LIST(),
          LIST(),
          LIST(3, 2),
          LIST(0, 0, 1, 0, 2, 0),
          LIST(7.0, 7.0, 7.0),
          LIST(1, 1, 1),
          "AllEmptyRows");

TEST_DATA(LIST(1, 0, 1, 1),
          LIST(5, 6),
          LIST(3, 3),
          LIST(0, 0, 1, 0, 1, 1, 2, 0),
          LIST(-1.0, 5.0, 6.0, -1.0),
          LIST(1, 0, 1),
          "EmptyRowsAtBothEnds");

TEST_DATA(LIST(0, 2, 2, 0, 4, 1, 6, 3),
          LIST(1.5, 2.5, 3.5, 4.5),
          LIST(8, 4),
          LIST(0, 2, 1, 0, 2, 0, 3, 0, 4, 1, 5, 0, 6, 3, 7, 0),
          LIST(1.5, 0.5, 2.5, 0.5, 3.5, 0.5, 4.5, 0.5),
          LIST(0, 1, 0, 1, 0, 1, 0, 1),
          "LargerDimensions");