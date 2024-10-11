#pragma once

#define LIST(...) \
    { __VA_ARGS__ }

// TEST_DATA(sorted_shape,
//           values_shape,
//           right_mode,
//           sorted_data,
//           values_data,
//           expected_output_data,
//           description)

// NOTE: expected output were generated using pyTorch.searchsorted implementation.

TEST_DATA(LIST(5),
          LIST(2, 3),
          false,
          LIST(1, 3, 5, 7, 9),
          LIST(3, 6, 9, 3, 6, 9),
          LIST(1, 3, 4, 1, 3, 4),
          "1d_tensor_1");

TEST_DATA(LIST(5),
          LIST(4, 3),
          false,
          LIST(1, 3, 5, 7, 9),
          LIST(0, 6, 20, 1, 6, 9, 1, 0, 0, 9, 10, 20),
          LIST(0, 3, 5, 0, 3, 4, 0, 0, 0, 4, 5, 5),
          "1d_tensor_2");

TEST_DATA(LIST(5),
          LIST(4, 3),
          true,
          LIST(1, 3, 5, 7, 9),
          LIST(0, 6, 20, 1, 6, 9, 1, 0, 0, 9, 10, 20),
          LIST(0, 3, 5, 1, 3, 5, 1, 0, 0, 5, 5, 5),
          "1d_tensor_2_right_mode");

TEST_DATA(LIST(5),
          LIST(2, 2, 3),
          false,
          LIST(1, 3, 5, 7, 9),
          LIST(0, 6, 20, 1, 6, 9, 1, 0, 0, 9, 10, 20),
          LIST(0, 3, 5, 0, 3, 4, 0, 0, 0, 4, 5, 5),
          "1d_tensor_3");

TEST_DATA(LIST(5),
          LIST(2, 2, 3),
          true,
          LIST(1, 3, 5, 7, 9),
          LIST(0, 6, 20, 1, 6, 9, 1, 0, 0, 9, 10, 20),
          LIST(0, 3, 5, 1, 3, 5, 1, 0, 0, 5, 5, 5),
          "1d_tensor_3_right_mode");

TEST_DATA(LIST(2, 5),
          LIST(2, 3),
          false,
          LIST(1, 3, 5, 7, 9, 2, 4, 6, 8, 10),
          LIST(3, 6, 9, 3, 6, 9),
          LIST(1, 3, 4, 1, 2, 4),
          "nd_tensor_1");

TEST_DATA(LIST(2, 5),
          LIST(2, 3),
          true,
          LIST(1, 3, 5, 7, 9, 2, 4, 6, 8, 10),
          LIST(3, 6, 9, 3, 6, 9),
          LIST(2, 3, 5, 1, 3, 4),
          "nd_tensor_1_right_mode");

TEST_DATA(LIST(2, 2, 5),
          LIST(2, 2, 3),
          false,
          LIST(1, 3, 5, 7, 9, 0, 2, 4, 6, 8, -20, 5, 10, 23, 41, 100, 125, 130, 132, 139),
          LIST(0, 6, 20, 1, 6, 9, 1, 0, 0, 9, 10, 20),
          LIST(0, 3, 5, 1, 3, 5, 1, 1, 1, 0, 0, 0),
          "nd_tensor_2");

TEST_DATA(LIST(2, 2, 5),
          LIST(2, 2, 3),
          true,
          LIST(1, 3, 5, 7, 9, 0, 2, 4, 6, 8, -20, 5, 10, 23, 41, 100, 125, 130, 132, 139),
          LIST(0, 6, 20, 1, 6, 9, 1, 0, 0, 9, 10, 20),
          LIST(0, 3, 5, 1, 4, 5, 1, 1, 1, 0, 0, 0),
          "nd_tensor_2");