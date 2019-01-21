// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "memory_solver.hpp"
#include "details/ie_exception.hpp"

using namespace testing;
using namespace InferenceEngine;
using Box = InferenceEngine::MemorySolver::Box;

TEST(MemSolverTest, LinearAndEven) {
    int n = 0;
    std::vector<Box> boxes {  //  |
            {n, ++n, 2},      //  |      ____
            {n, ++n, 2},      //  |   __|____|__
            {n, ++n, 2},      //  |__|____||____|__
    };                        //      0  1  2  3

    MemorySolver ms(boxes);
    EXPECT_EQ(ms.solve(), 4);
    EXPECT_EQ(ms.maxDepth(), 4);
    EXPECT_EQ(ms.maxTopDepth(), 2);
}

TEST(MemSolverTest, LinearAndNotEven) {
    int n = 0;
    std::vector<Box> boxes {  //  |      ____
            {n, ++n, 2},      //  |     |____|__
            {n, ++n, 2},      //  |   ____ |    |
            {n, ++n, 3},      //  |__|____||____|__
    };                        //      0  1  2  3

    MemorySolver ms(boxes);
    EXPECT_EQ(ms.solve(), 5);
    EXPECT_EQ(ms.maxDepth(), 5);
    EXPECT_EQ(ms.maxTopDepth(), 2);
}


TEST(MemSolverTest, LinearWithEmptyExecIndexes) {
    int n = 2;
    std::vector<Box> boxes {   //  |         _______
            {n, n+=2, 2},      //  |        |_______|_____
            {n, n+=2, 2},      //  |   _______    |       |
            {n, n+=2, 3},      //  |__|_______|___|_______|__
    };                         //      2  3  4  5  6  7  8

    MemorySolver ms(boxes);
    EXPECT_EQ(ms.solve(), 5);
    EXPECT_EQ(ms.maxDepth(), 5);
    EXPECT_EQ(ms.maxTopDepth(), 2);
}

TEST(MemSolverTest, DISABLED_Unefficiency) {

    std::vector<Box> boxes{    //  |            __________
            {6, 7, 3},         //  |   ____    |_3________|
            {2, 5, 2},         //  |  |_4__|_____ |    |
            {5, 8, 2},         //  |__|_2________||_1__|___
            {2, 3, 2},         //      2  3  4  5  6  7  8
    };

    MemorySolver ms(boxes);
    EXPECT_EQ(ms.solve(), 5); // currently we have answer 6
    EXPECT_EQ(ms.maxDepth(), 5);
    EXPECT_EQ(ms.maxTopDepth(), 2);
}

TEST(MemSolverTest, OverlappingBoxes) {

    std::vector<Box> boxes{    //  |            __________
            {6, 7, 4},         //  |   ____    |_3________|
            {2, 5, 3},         //  |  |_4__|_____ |    |
            {5, 8, 2},         //  |__|_2________||_1__|___
            {2, 3, 2},         //      2  3  4  5  6  7  8
    };

    MemorySolver ms(boxes);
    EXPECT_EQ(ms.solve(), 6);
    EXPECT_EQ(ms.maxDepth(), 6);
    EXPECT_EQ(ms.maxTopDepth(), 2);
}

TEST(MemSolverTest, EndOnSeveralBegins) {

    std::vector<Box> boxes {   //  |      ____
            {0, 1, 2},         //  |     |____| ____
            {1, 2, 2},         //  |           |____|__
            {3, 3, 2},         //  |   ____    |_______|
            {3, 5, 2},         //  |__|____|___|_|_________
            {3, 4, 2},         //      0  1  2  3  4  5  6
    };

    MemorySolver ms(boxes);
    EXPECT_EQ(ms.solve(), 6);
    EXPECT_EQ(ms.maxDepth(), 6);
    EXPECT_EQ(ms.maxTopDepth(), 3);
}

TEST(MemSolverTest, ToEndBoxes) {

    std::vector<Box> boxes {   //  |      _____________
            {0, 1, 2},         //  |     |_____________>>
            {1,-1, 2},         //  |           |____|__
            {3, 3, 2},         //  |   ____    |_______>>
            {3,-1, 2},         //  |__|____|___|_|_________
            {3, 4, 2},         //      0  1  2  3  4  5  6
    };

    MemorySolver ms(boxes);
    EXPECT_EQ(ms.solve(), 8);
    EXPECT_EQ(ms.maxDepth(), 8);
    EXPECT_EQ(ms.maxTopDepth(), 4);
}

TEST(MemSolverTest, LastAndToEndBox) {

    std::vector<Box> boxes {   //  |                     _
            {0, 1, 2},         //  |            ____    |_>>
            {6,-1, 2},         //  |           |____|__
            {3, 3, 2},         //  |   ____    |_______|
            {3, 5, 2},         //  |__|____|___|_|_________
            {3, 4, 2},         //      0  1  2  3  4  5  6
    };

    MemorySolver ms(boxes);
    EXPECT_EQ(ms.solve(), 6);
    EXPECT_EQ(ms.maxDepth(), 6);
    EXPECT_EQ(ms.maxTopDepth(), 3);
}

TEST(MemSolverTest, OptimalAlexnet) {
    std::vector<std::vector<int>> shapes {
            {3,227,227},  // in
            {96 ,55,55},  // conv1
            {96 ,55,55},  // relu1
            {96 ,55,55},  // norm1
            {96 ,27,27},  // pool1
            {256,27,27},  // conv2
            {256,27,27},  // relu2
            {256,27,27},  // norm2
            {256,13,13},  // pool2
            {384,13,13},  // conv3
            {384,13,13},  // relu3
            {384,13,13},  // conv4
            {384,13,13},  // relu4
            {256,13,13},  // conv5
            {256,13,13},  // relu5
            {256, 6, 6},  // pool5
            {1,1 ,4069},  // fc6
            {1,1 ,4069},  // relu6
            {1,1 ,4069},  // fc7
            {1,1 ,4069},  // relu7
            {1,1 ,1000},  // fc8
            {1,1 ,1000},  // loss
    };

    int n = 0;
    std::vector<Box> boxes;
    for (const auto &sh : shapes) boxes.push_back( {n, ++n, sh[0]*sh[1]*sh[2]} );

    // For linear topology bottom score is reachable minRequired == maxDepth
    MemorySolver ms(boxes);
    EXPECT_EQ(ms.solve(), ms.maxDepth());
    EXPECT_EQ(ms.maxTopDepth(), 2);
}

TEST(MemSolverTest, GetOffsets) {
    int n = 0;
    std::vector<Box> boxes{   //  |
            {n, ++n, 2, 0},   //  |      ____  ____
            {n, ++n, 2, 1},   //  |   __|____||____|
            {n, ++n, 2, 2},   //  |__|____||____|_____
            {n, ++n, 2, 3},   //      0  1  2  3  4
    };

    MemorySolver ms(boxes);
    ms.solve();

    //  The correct answer is [0, 2, 0, 2] or [2, 0, 2, 0].
    EXPECT_EQ(ms.getOffset(0) + ms.getOffset(1), 2);
    EXPECT_EQ(ms.getOffset(1) + ms.getOffset(2), 2);
    EXPECT_EQ(ms.getOffset(2) + ms.getOffset(3), 2);
}

TEST(MemSolverTest, GetOffsetThows) {
    int n = 0, id = 0;
    std::vector<Box> boxes{      //  |
            {n, ++n, 2, id++},   //  |      ____  ____
            {n, ++n, 2, id++},   //  |   __|____||____|
            {n, ++n, 2, id++},   //  |__|____||____|_____
            {n, ++n, 2, id++},   //      0  1  2  3  4
    };

    MemorySolver ms(boxes);
    ms.solve();

    EXPECT_THROW(ms.getOffset(100), details::InferenceEngineException);
}

TEST(MemSolverTest, NoOverlapping) {

    int n = 0;                //  |         _____________
    std::vector<Box> boxes{   //  |   _____|___1_________|
            {4, 8, 1, n++},   //  |  |_2_____|    ____
            {6, 7, 3, n++},   //  |  |    |      |    |
            {2, 3, 3, n++},   //  |__|_3__|______|_3__|___
            {2, 4, 2, n++},   //      2  3  4  5  6  7  8
    };

    MemorySolver ms(boxes);
    ms.solve();
    // TODO: Current algorithm doesn't solve that case. Uncomment check to see inefficiency
    // EXPECT_EQ(ms.solve(), 5);

    auto no_overlap = [&](Box box1, Box box2) -> bool {
        int off1 = ms.getOffset(box1.id);
        int off2 = ms.getOffset(box2.id);
        return box1.finish < box2.start || box1.start > box2.finish ||
               off1 + box1.size <= off2 || off1 >= off2 + box2.size;
    };

    for (int i = 0; i < n; i++)
    for (int j = i+1; j < n; j++)
        ASSERT_TRUE(no_overlap(boxes[i], boxes[j])) << "Box overlapping is detected";
}

TEST(MemSolverTest, BestSolution1) {

    int n = 0;                //  |         _______
    std::vector<Box> boxes{   //  |        |_2_____|__
            {2, 3, 1, n++},   //  |      ____    |    |
            {3, 4, 1, n++},   //  |   __|_1__|   |    |
            {4, 6, 2, n++},   //  |__|_1__|______|_3__|___
            {6, 7, 3, n++},   //      2  3  4  5  6  7  8
    };

    MemorySolver ms(boxes);
    EXPECT_EQ(ms.solve(), 5);

    auto no_overlap = [&](Box box1, Box box2) -> bool {
        int off1 = ms.getOffset(box1.id);
        int off2 = ms.getOffset(box2.id);
        return box1.finish < box2.start || box1.start > box2.finish ||
               off1 + box1.size <= off2 || off1 >= off2 + box2.size;
    };

    for (int i = 0; i < n; i++)
        for (int j = i+1; j < n; j++)
            ASSERT_TRUE(no_overlap(boxes[i], boxes[j])) << "Box overlapping is detected";
}

