// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/memory_solver.hpp"

#include <gtest/gtest.h>

#include <vector>

using Box = ov::MemorySolver::Box;

TEST(MemSolverTest, CanConstruct) {
    {  // Empty vector<Box>
        ov::MemorySolver ms(std::vector<Box>{});
    }

    {  // vector with default Box
        ov::MemorySolver ms(std::vector<Box>{{}});
    }

    {  // vector with Box with non-default Box
        ov::MemorySolver ms(std::vector<Box>{{1, 3, 3}});
    }

    {  // vector with Box with size == 0
        ov::MemorySolver ms(std::vector<Box>{{0, 0, 0}});
    }

    {  // vector with Box with finish == -1
        ov::MemorySolver ms(std::vector<Box>{{3, -1, 6}});
    }

    // TODO: enable after implement TODO from memory_solver.hpp#L66
    //    {   // vector with Box with negative values
    //        MemorySolver ms(std::vector<Box> {{-5, -5, -5, -5}});
    //    }
}

//  |
//  |      ____  ____
//  |   __|____||____|
//  |__|____||____|_____
//      0  1  2  3  4
TEST(MemSolverTest, get_offset) {
    int n = 0;
    std::vector<Box> boxes{
        {n, ++n, 2, 0},
        {n, ++n, 2, 1},
        {n, ++n, 2, 2},
        {n, ++n, 2, 3},
    };

    ov::MemorySolver ms(boxes);
    ms.solve();

    //  The correct answer is [0, 2, 0, 2] or [2, 0, 2, 0].
    EXPECT_EQ(ms.get_offset(0) + ms.get_offset(1), 2);
    EXPECT_EQ(ms.get_offset(1) + ms.get_offset(2), 2);
    EXPECT_EQ(ms.get_offset(2) + ms.get_offset(3), 2);
}

//  |
//  |      ____  ____
//  |   __|____||____|
//  |__|____||____|_____
//      0  1  2  3  4
TEST(MemSolverTest, get_offsetThrowException) {
    int n = 0, id = 0;
    std::vector<Box> boxes{
        {n, ++n, 2, id++},
        {n, ++n, 2, id++},
        {n, ++n, 2, id++},
        {n, ++n, 2, id++},
    };

    ov::MemorySolver ms(boxes);
    ms.solve();

    EXPECT_THROW(ms.get_offset(100), std::runtime_error);
}

//  |
//  |      ____
//  |   __|____|__
//  |__|____||____|__
//      0  1  2  3
TEST(MemSolverTest, LinearAndEven) {
    int n = 0;
    std::vector<Box> boxes{
        {n, ++n, 2},
        {n, ++n, 2},
        {n, ++n, 2},
    };

    ov::MemorySolver ms(boxes);
    EXPECT_EQ(ms.solve(), 4);
    EXPECT_EQ(ms.max_depth(), 4);
    EXPECT_EQ(ms.max_top_depth(), 2);
}

//  |      ____
//  |     |____|__
//  |   ____ |    |
//  |__|____||____|__
//      0  1  2  3
TEST(MemSolverTest, LinearAndNotEven) {
    int n = 0;
    std::vector<Box> boxes{
        {n, ++n, 2},
        {n, ++n, 2},
        {n, ++n, 3},
    };

    ov::MemorySolver ms(boxes);
    EXPECT_EQ(ms.solve(), 5);
    EXPECT_EQ(ms.max_depth(), 5);
    EXPECT_EQ(ms.max_top_depth(), 2);
}

//  |         _______
//  |        |_______|_____
//  |   _______    |       |
//  |__|_______|___|_______|__
//      2  3  4  5  6  7  8
TEST(MemSolverTest, LinearWithEmptyExecIndexes) {
    int n = 2;
    std::vector<Box> boxes{
        {n, n += 2, 2},
        {n, n += 2, 2},
        {n, n += 2, 3},
    };

    ov::MemorySolver ms(boxes);
    EXPECT_EQ(ms.solve(), 5);
    EXPECT_EQ(ms.max_depth(), 5);
    EXPECT_EQ(ms.max_top_depth(), 2);
}

//  |            __________
//  |   ____    |_3________|
//  |  |_4__|_____ |    |
//  |__|_2________||_1__|___
//      2  3  4  5  6  7  8
TEST(MemSolverTest, DISABLED_Unefficiency) {
    std::vector<Box> boxes{
        {6, 7, 3},
        {2, 5, 2},
        {5, 8, 2},
        {2, 3, 2},
    };

    ov::MemorySolver ms(boxes);
    EXPECT_EQ(ms.solve(), 5);  // currently we have answer 6
    EXPECT_EQ(ms.max_depth(), 5);
    EXPECT_EQ(ms.max_top_depth(), 2);
}

//  |            __________
//  |   ____    |_3________|
//  |  |_4__|_____ |    |
//  |__|_2________||_1__|___
//      2  3  4  5  6  7  8
TEST(MemSolverTest, OverlappingBoxes) {
    std::vector<Box> boxes{
        {6, 7, 4},
        {2, 5, 3},
        {5, 8, 2},
        {2, 3, 2},
    };

    ov::MemorySolver ms(boxes);
    EXPECT_EQ(ms.solve(), 6);
    EXPECT_EQ(ms.max_depth(), 6);
    EXPECT_EQ(ms.max_top_depth(), 2);
}

//  |      ____
//  |     |____| ____
//  |           |____|__
//  |   ____    |_______|
//  |__|____|___|_|_________
//      0  1  2  3  4  5  6
TEST(MemSolverTest, EndOnSeveralBegins) {
    std::vector<Box> boxes{
        {0, 1, 2},
        {1, 2, 2},
        {3, 3, 2},
        {3, 5, 2},
        {3, 4, 2},
    };

    ov::MemorySolver ms(boxes);
    EXPECT_EQ(ms.solve(), 6);
    EXPECT_EQ(ms.max_depth(), 6);
    EXPECT_EQ(ms.max_top_depth(), 3);
}

//  |      _____________
//  |     |_____________>>
//  |           |____|__
//  |   ____    |_______>>
//  |__|____|___|_|_________
//      0  1  2  3  4  5  6
TEST(MemSolverTest, ToEndBoxes) {
    std::vector<Box> boxes{
        {0, 1, 2},
        {1, -1, 2},
        {3, 3, 2},
        {3, -1, 2},
        {3, 4, 2},
    };

    ov::MemorySolver ms(boxes);
    EXPECT_EQ(ms.solve(), 8);
    EXPECT_EQ(ms.max_depth(), 8);
    EXPECT_EQ(ms.max_top_depth(), 4);
}

//  |                     _
//  |            ____    |_>>
//  |           |____|__
//  |   ____    |_______|
//  |__|____|___|_|_________
//      0  1  2  3  4  5  6
TEST(MemSolverTest, LastAndToEndBox) {
    std::vector<Box> boxes{
        {0, 1, 2},
        {6, -1, 2},
        {3, 3, 2},
        {3, 5, 2},
        {3, 4, 2},
    };

    ov::MemorySolver ms(boxes);
    EXPECT_EQ(ms.solve(), 6);
    EXPECT_EQ(ms.max_depth(), 6);
    EXPECT_EQ(ms.max_top_depth(), 3);
}

TEST(MemSolverTest, OptimalAlexnet) {
    std::vector<std::vector<int>> shapes{
        {3, 227, 227},  // in
        {96, 55, 55},   // conv1
        {96, 55, 55},   // relu1
        {96, 55, 55},   // norm1
        {96, 27, 27},   // pool1
        {256, 27, 27},  // conv2
        {256, 27, 27},  // relu2
        {256, 27, 27},  // norm2
        {256, 13, 13},  // pool2
        {384, 13, 13},  // conv3
        {384, 13, 13},  // relu3
        {384, 13, 13},  // conv4
        {384, 13, 13},  // relu4
        {256, 13, 13},  // conv5
        {256, 13, 13},  // relu5
        {256, 6, 6},    // pool5
        {1, 1, 4069},   // fc6
        {1, 1, 4069},   // relu6
        {1, 1, 4069},   // fc7
        {1, 1, 4069},   // relu7
        {1, 1, 1000},   // fc8
        {1, 1, 1000},   // loss
    };

    int n = 0;
    std::vector<Box> boxes;
    for (const auto& sh : shapes)
        boxes.push_back({n, ++n, sh[0] * sh[1] * sh[2]});

    // For linear topology bottom score is reachable minRequired == max_depth
    ov::MemorySolver ms(boxes);
    EXPECT_EQ(ms.solve(), ms.max_depth());
    EXPECT_EQ(ms.max_top_depth(), 2);
}

//  |         _____________
//  |   _____|___1_________|
//  |  |_2_____|    ____
//  |  |    |      |    |
//  |__|_3__|______|_3__|___
//      2  3  4  5  6  7  8
TEST(MemSolverTest, NoOverlapping) {
    int n = 0;
    std::vector<Box> boxes{
        {4, 8, 1, n++},
        {6, 7, 3, n++},
        {2, 3, 3, n++},
        {2, 4, 2, n++},
    };

    ov::MemorySolver ms(boxes);
    ms.solve();
    // TODO: Current algorithm doesn't solve that case. Uncomment check to see inefficiency
    // EXPECT_EQ(ms.solve(), 5);

    auto no_overlap = [&](Box box1, Box box2) -> bool {
        int64_t off1 = ms.get_offset(static_cast<int>(box1.id));
        int64_t off2 = ms.get_offset(static_cast<int>(box2.id));
        return box1.finish < box2.start || box1.start > box2.finish || off1 + box1.size <= off2 ||
               off1 >= off2 + box2.size;
    };

    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            ASSERT_TRUE(no_overlap(boxes[i], boxes[j])) << "Box overlapping is detected";
}

//  |         _______
//  |        |_2_____|__
//  |      ____    |    |
//  |   __|_1__|   |    |
//  |__|_1__|______|_3__|___
//      2  3  4  5  6  7  8
TEST(MemSolverTest, BestSolution1) {
    int n = 0;
    std::vector<Box> boxes{
        {2, 3, 1, n++},
        {3, 4, 1, n++},
        {4, 6, 2, n++},
        {6, 7, 3, n++},
    };

    ov::MemorySolver ms(boxes);
    EXPECT_EQ(ms.solve(), 5);

    auto no_overlap = [&](Box box1, Box box2) -> bool {
        int64_t off1 = ms.get_offset(static_cast<int>(box1.id));
        int64_t off2 = ms.get_offset(static_cast<int>(box2.id));
        return box1.finish < box2.start || box1.start > box2.finish || off1 + box1.size <= off2 ||
               off1 >= off2 + box2.size;
    };

    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            ASSERT_TRUE(no_overlap(boxes[i], boxes[j])) << "Box overlapping is detected";
}
