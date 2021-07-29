// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/axis_vector.hpp"
#include "ngraph/runtime/opt_kernel/reshape.hpp"
#include "ngraph/shape.hpp"

#include "util/ndarray.hpp"

using namespace ngraph;

namespace
{
    using ElementValue = int32_t;
    enum class AxisOrder
    {
        straight,
        reverse,
    };

    AxisVector get_axis_order(AxisOrder order, size_t size)
    {
        AxisVector v(size);
        std::iota(begin(v), end(v), 0);
        if (order == AxisOrder::reverse)
        {
            std::reverse(begin(v), end(v));
        }
        return v;
    }

    struct TestParams
    {
        AxisOrder order;
        test::NDArrayBase<ElementValue> input;
        test::NDArrayBase<ElementValue> output;
    };

    struct ReshapeOptKernel : ::testing::TestWithParam<TestParams>
    {
    };

} // namespace

TEST_P(ReshapeOptKernel, reshape_opt_kernel)
{
    const TestParams& p = GetParam();

    const AxisVector axis_order = get_axis_order(p.order, p.input.get_shape().size());
    std::vector<ElementValue> output_buff(p.input.get_vector().size());

    runtime::opt_kernel::reshape((const char*)p.input.data(),
                                 (char*)output_buff.data(),
                                 p.input.get_shape(),
                                 axis_order,
                                 p.output.get_shape(),
                                 sizeof(ElementValue));
    EXPECT_EQ(p.output.get_vector(), output_buff);
}

INSTANTIATE_TEST_SUITE_P(reshape_opt_kernel,
                         ReshapeOptKernel,
                         ::testing::Values(TestParams{AxisOrder::straight,
                                                      test::NDArray<ElementValue, 2>{
                                                          {1, 2},
                                                          {3, 4},
                                                          {5, 6},
                                                      },
                                                      test::NDArray<ElementValue, 2>{
                                                          {1, 2, 3},
                                                          {4, 5, 6},
                                                      }},
                                           TestParams{AxisOrder::straight,
                                                      test::NDArray<ElementValue, 2>{
                                                          {1, 2},
                                                          {3, 4},
                                                          {5, 6},
                                                      },
                                                      test::NDArray<ElementValue, 1>{
                                                          {1, 2, 3, 4, 5, 6},
                                                      }},
                                           TestParams{AxisOrder::straight,
                                                      test::NDArray<ElementValue, 3>{
                                                          {
                                                              {11, 12},
                                                              {13, 14},
                                                              {15, 16},
                                                          },
                                                          {
                                                              {21, 22},
                                                              {23, 24},
                                                              {25, 26},
                                                          },
                                                      },
                                                      test::NDArray<ElementValue, 2>{
                                                          {11, 12, 13, 14, 15, 16},
                                                          {21, 22, 23, 24, 25, 26},
                                                      }},
                                           TestParams{AxisOrder::straight,
                                                      test::NDArray<ElementValue, 4>{
                                                          {
                                                              {
                                                                  {11, 12},
                                                                  {13, 14},
                                                                  {15, 16},
                                                              },
                                                              {
                                                                  {21, 22},
                                                                  {23, 24},
                                                                  {25, 26},
                                                              },
                                                          },
                                                      },
                                                      test::NDArray<ElementValue, 2>{
                                                          {11, 12, 13, 14, 15, 16},
                                                          {21, 22, 23, 24, 25, 26},
                                                      }},
                                           TestParams{AxisOrder::reverse,
                                                      test::NDArray<ElementValue, 2>{
                                                          {1, 2},
                                                          {3, 4},
                                                          {5, 6},
                                                      },
                                                      test::NDArray<ElementValue, 2>{
                                                          {1, 3, 5},
                                                          {2, 4, 6},
                                                      }},
                                           TestParams{AxisOrder::reverse,
                                                      test::NDArray<ElementValue, 2>{
                                                          {1, 2},
                                                          {3, 4},
                                                          {5, 6},
                                                      },
                                                      test::NDArray<ElementValue, 1>{
                                                          {1, 3, 5, 2, 4, 6},
                                                      }},
                                           TestParams{AxisOrder::reverse,
                                                      test::NDArray<ElementValue, 3>{
                                                          {
                                                              {11, 12},
                                                              {13, 14},
                                                              {15, 16},
                                                          },
                                                          {
                                                              {21, 22},
                                                              {23, 24},
                                                              {25, 26},
                                                          },
                                                      },
                                                      test::NDArray<ElementValue, 2>{
                                                          {11, 21, 13, 23, 15, 25},
                                                          {12, 22, 14, 24, 16, 26},
                                                      }},
                                           TestParams{AxisOrder::reverse,
                                                      test::NDArray<ElementValue, 4>{
                                                          {
                                                              {
                                                                  {11, 12},
                                                                  {13, 14},
                                                                  {15, 16},
                                                              },
                                                              {
                                                                  {21, 22},
                                                                  {23, 24},
                                                                  {25, 26},
                                                              },
                                                          },
                                                      },
                                                      test::NDArray<ElementValue, 2>{
                                                          {11, 21, 13, 23, 15, 25},
                                                          {12, 22, 14, 24, 16, 26},
                                                      }}));

// input shape with size > 6 should be covered by reference implementation:
INSTANTIATE_TEST_SUITE_P(reshape_opt_kernel_ref_impl_fallback,
                         ReshapeOptKernel,
                         ::testing::Values(TestParams{AxisOrder::straight,
                                                      test::NDArray<ElementValue, 7>{
                                                          {
                                                              {
                                                                  {
                                                                      {
                                                                          {
                                                                              {11, 12},
                                                                              {13, 14},
                                                                              {15, 16},
                                                                          },
                                                                          {
                                                                              {21, 22},
                                                                              {23, 24},
                                                                              {25, 26},
                                                                          },
                                                                      },
                                                                  },
                                                              },
                                                          },
                                                      },
                                                      test::NDArray<ElementValue, 2>{
                                                          {11, 12, 13, 14, 15, 16},
                                                          {21, 22, 23, 24, 25, 26},
                                                      }},
                                           TestParams{AxisOrder::reverse,
                                                      test::NDArray<ElementValue, 7>{
                                                          {
                                                              {
                                                                  {
                                                                      {
                                                                          {
                                                                              {11, 12},
                                                                              {13, 14},
                                                                              {15, 16},
                                                                          },
                                                                          {
                                                                              {21, 22},
                                                                              {23, 24},
                                                                              {25, 26},
                                                                          },
                                                                      },
                                                                  },
                                                              },
                                                          },
                                                      },
                                                      test::NDArray<ElementValue, 2>{
                                                          {11, 21, 13, 23, 15, 25},
                                                          {12, 22, 14, 24, 16, 26},
                                                      }}));
