// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/slice_scatter.hpp>
#include <intel_gpu/primitives/data.hpp>

#include <vector>
#include <numeric>
#include <chrono>

using namespace cldnn;
using namespace ::tests;

namespace {

namespace helpers {

template<typename T>
data_types ToDataType();

template<>
data_types ToDataType<float>() { return data_types::f32; }

template<>
data_types ToDataType<int32_t>() { return data_types::i32; }

template<>
data_types ToDataType<int64_t>() { return data_types::i64; }

template<typename T>
std::vector<T> GenInput(const ov::PartialShape& shape) {
    const size_t size = ov::shape_size(shape.get_shape());
    std::vector<T> result(size);
    std::iota(result.begin(), result.end(), T{0});
    return result;
}

} // namespace helpers

struct SliceScatterTestParams {
    memory::ptr data;
    memory::ptr updates;
    memory::ptr start;
    memory::ptr stop;
    memory::ptr step;
    memory::ptr axes;
    memory::ptr wanted_output;
    bool is_data_dynamic = false;
    bool is_updates_dynamic = false;
    bool is_start_dynamic = false;
    bool is_stop_dynamic = false;
    bool is_step_dynamic = false;
    bool is_axes_dynamic = false;
    bool is_caching_test = false;
};

template<typename T>
class SliceScatterTest : public ::testing::Test {
public:
    void RunAllTestCasesForParams(const SliceScatterTestParams& params) {
        RunTestCase(params);
    }

    template<typename TDataType>
    memory::ptr AllocateTensor(ov::PartialShape shape, cldnn::format fmt,
                                const std::vector<TDataType>& data) {
        const layout lo = {shape, helpers::ToDataType<TDataType>(), fmt};
        EXPECT_EQ(lo.get_linear_size(), data.size());
        memory::ptr tensor = this->engine_.allocate_memory(lo);
        set_values<TDataType>(tensor, data);
        return tensor;
    }

    // Test case: basic bfyx with positive step, no axes
    // data [1,1,3,4] = GenInput(0..11)
    // updates [1,1,2,2] = [20,21,22,23]
    // start=[0,0,0,1], stop=[1,1,2,3], step=[1,1,1,1]
    // Slice region: b[0:1],f[0:1],y[0:2],x[1:3]
    // Expected: [0,20,21,3, 4,22,23,7, 8,9,10,11]
    template<typename TypeParam>
    void FillWithBasicBfyxData(SliceScatterTestParams& params) {
        const ov::PartialShape data_shape{1, 1, 3, 4};
        params.data = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfyx, helpers::GenInput<TypeParam>(data_shape));
        params.updates = this->template AllocateTensor<TypeParam>(
            ov::PartialShape{1, 1, 2, 2}, format::bfyx,
            std::vector<TypeParam>{20, 21, 22, 23});
        params.start = this->template AllocateTensor<int64_t>(
            ov::PartialShape{4}, format::bfyx, {0, 0, 0, 1});
        params.stop = this->template AllocateTensor<int64_t>(
            ov::PartialShape{4}, format::bfyx, {1, 1, 2, 3});
        params.step = this->template AllocateTensor<int64_t>(
            ov::PartialShape{4}, format::bfyx, {1, 1, 1, 1});
        params.wanted_output = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfyx,
            std::vector<TypeParam>{0, 20, 21, 3, 4, 22, 23, 7, 8, 9, 10, 11});
    }

    // Test case: with explicit axes parameter
    // data [1,1,4,4] = GenInput(0..15)
    // updates [1,1,2,4] = [100..107]
    // start=[1], stop=[3], step=[1], axes=[2]
    // Slice axis 2 (y): y[1:3], all other axes full
    // Expected: [0,1,2,3, 100,101,102,103, 104,105,106,107, 12,13,14,15]
    template<typename TypeParam>
    void FillWithAxesData(SliceScatterTestParams& params) {
        const ov::PartialShape data_shape{1, 1, 4, 4};
        params.data = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfyx, helpers::GenInput<TypeParam>(data_shape));
        params.updates = this->template AllocateTensor<TypeParam>(
            ov::PartialShape{1, 1, 2, 4}, format::bfyx,
            std::vector<TypeParam>{100, 101, 102, 103, 104, 105, 106, 107});
        params.start = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {1});
        params.stop = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {3});
        params.step = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {1});
        params.axes = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {2});
        params.wanted_output = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfyx,
            std::vector<TypeParam>{0, 1, 2, 3, 100, 101, 102, 103,
                                   104, 105, 106, 107, 12, 13, 14, 15});
    }

    // Test case: step > 1
    // data [1,1,1,8] = GenInput(0..7)
    // updates [1,1,1,4] = [10,11,12,13]
    // start=[0], stop=[8], step=[2], axes=[3]
    // Scatter at x=0,2,4,6
    // Expected: [10,1,11,3,12,5,13,7]
    template<typename TypeParam>
    void FillWithStepData(SliceScatterTestParams& params) {
        const ov::PartialShape data_shape{1, 1, 1, 8};
        params.data = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfyx, helpers::GenInput<TypeParam>(data_shape));
        params.updates = this->template AllocateTensor<TypeParam>(
            ov::PartialShape{1, 1, 1, 4}, format::bfyx,
            std::vector<TypeParam>{10, 11, 12, 13});
        params.start = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {0});
        params.stop = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {8});
        params.step = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {2});
        params.axes = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {3});
        params.wanted_output = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfyx,
            std::vector<TypeParam>{10, 1, 11, 3, 12, 5, 13, 7});
    }

    // Test case: negative indices
    // data [1,1,1,8] = GenInput(0..7)
    // updates [1,1,1,2] = [10,11]
    // start=[-3], stop=[-1], step=[1], axes=[3]
    // start=-3 -> 5, stop=-1 -> 7: x[5:7] = [10,11]
    // Expected: [0,1,2,3,4,10,11,7]
    template<typename TypeParam>
    void FillWithNegativeIndicesData(SliceScatterTestParams& params) {
        const ov::PartialShape data_shape{1, 1, 1, 8};
        params.data = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfyx, helpers::GenInput<TypeParam>(data_shape));
        params.updates = this->template AllocateTensor<TypeParam>(
            ov::PartialShape{1, 1, 1, 2}, format::bfyx,
            std::vector<TypeParam>{10, 11});
        params.start = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {-3});
        params.stop = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {-1});
        params.step = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {1});
        params.axes = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {3});
        params.wanted_output = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfyx,
            std::vector<TypeParam>{0, 1, 2, 3, 4, 10, 11, 7});
    }

    // Test case: 5D bfzyx
    // data [1,2,2,2,3] = GenInput(0..23)
    // updates [1,1,1,2,3] = [100..105]
    // start=[0,0,0,0,0], stop=[1,1,1,2,3], step=[1,1,1,1,1]
    // Replace b=0,f=0,z=0 region
    // Expected: [100,101,102,103,104,105, 6..23]
    template<typename TypeParam>
    void FillWithBfzyxData(SliceScatterTestParams& params) {
        const ov::PartialShape data_shape{1, 2, 2, 2, 3};
        params.data = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfzyx, helpers::GenInput<TypeParam>(data_shape));
        params.updates = this->template AllocateTensor<TypeParam>(
            ov::PartialShape{1, 1, 1, 2, 3}, format::bfzyx,
            std::vector<TypeParam>{100, 101, 102, 103, 104, 105});
        params.start = this->template AllocateTensor<int64_t>(
            ov::PartialShape{5}, format::bfzyx, {0, 0, 0, 0, 0});
        params.stop = this->template AllocateTensor<int64_t>(
            ov::PartialShape{5}, format::bfzyx, {1, 1, 1, 2, 3});
        params.step = this->template AllocateTensor<int64_t>(
            ov::PartialShape{5}, format::bfzyx, {1, 1, 1, 1, 1});
        params.wanted_output = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfzyx,
            std::vector<TypeParam>{100, 101, 102, 103, 104, 105,
                                   6, 7, 8, 9, 10, 11,
                                   12, 13, 14, 15, 16, 17,
                                   18, 19, 20, 21, 22, 23});
    }

    // Test case: negative step (backward slicing)
    // data [1,1,1,8] = GenInput(0..7) = [0,1,2,3,4,5,6,7]
    // updates [1,1,1,3] = [10,11,12]
    // start=[6], stop=[0], step=[-2], axes=[3]
    // Backward: x=6,4,2 => positions 6,4,2
    // Expected: [0,1,12,3,11,5,10,7]
    template<typename TypeParam>
    void FillWithNegativeStepData(SliceScatterTestParams& params) {
        const ov::PartialShape data_shape{1, 1, 1, 8};
        params.data = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfyx, helpers::GenInput<TypeParam>(data_shape));
        params.updates = this->template AllocateTensor<TypeParam>(
            ov::PartialShape{1, 1, 1, 3}, format::bfyx,
            std::vector<TypeParam>{10, 11, 12});
        params.start = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {6});
        params.stop = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {0});
        params.step = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {-2});
        params.axes = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {3});
        params.wanted_output = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfyx,
            std::vector<TypeParam>{0, 1, 12, 3, 11, 5, 10, 7});
    }

    // Test case: start/stop clamping
    // Spec says: "A value larger than the size of a dimension is silently clamped."
    // data [1,1,2,5] = GenInput(0..9) = [[0,1,2,3,4],[5,6,7,8,9]]
    // updates [1,1,2,3] = [10,20,30,40,50,60]
    // start=[-25], stop=[25], step=[2], axes=[3]
    // start=-25 clamped to 0, stop=25 clamped to 5 => x=0,2,4
    // Expected: [[10,1,20,3,30],[40,6,50,8,60]]
    // This matches Spec Example 2 pattern (clamping + step=2)
    template<typename TypeParam>
    void FillWithClampingData(SliceScatterTestParams& params) {
        const ov::PartialShape data_shape{1, 1, 2, 5};
        params.data = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfyx, helpers::GenInput<TypeParam>(data_shape));
        params.updates = this->template AllocateTensor<TypeParam>(
            ov::PartialShape{1, 1, 2, 3}, format::bfyx,
            std::vector<TypeParam>{10, 20, 30, 40, 50, 60});
        params.start = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {-25});
        params.stop = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {25});
        params.step = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {2});
        params.axes = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {3});
        params.wanted_output = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfyx,
            std::vector<TypeParam>{10, 1, 20, 3, 30, 40, 6, 50, 8, 60});
    }

    // Test case: negative axes
    // data [1,1,4,4] = GenInput(0..15)
    // updates [1,1,2,4] = [100..107]
    // start=[1], stop=[3], step=[1], axes=[-2]
    // axes=-2 with rank=4 => axis 2 (y dimension)
    // Same result as with_axes (axis=2), y[1:3]
    // Expected: [0,1,2,3, 100,101,102,103, 104,105,106,107, 12,13,14,15]
    template<typename TypeParam>
    void FillWithNegativeAxesData(SliceScatterTestParams& params) {
        const ov::PartialShape data_shape{1, 1, 4, 4};
        params.data = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfyx, helpers::GenInput<TypeParam>(data_shape));
        params.updates = this->template AllocateTensor<TypeParam>(
            ov::PartialShape{1, 1, 2, 4}, format::bfyx,
            std::vector<TypeParam>{100, 101, 102, 103, 104, 105, 106, 107});
        params.start = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {1});
        params.stop = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {3});
        params.step = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {1});
        params.axes = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {-2});
        params.wanted_output = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfyx,
            std::vector<TypeParam>{0, 1, 2, 3, 100, 101, 102, 103,
                                   104, 105, 106, 107, 12, 13, 14, 15});
    }

    // Test case: Spec Example 1 - fill slice over axis==0
    // data [1,1,2,5] = [[0,1,2,3,4],[5,6,7,8,9]]
    // updates [1,1,1,5] = [[10,20,30,40,50]]
    // start=[0], stop=[1], step=[1], axes=[2]  (axis 0 in spec = axis 2 in bfyx)
    // y[0:1] replaced
    // Expected: [[10,20,30,40,50],[5,6,7,8,9]]
    template<typename TypeParam>
    void FillWithSpecExample1Data(SliceScatterTestParams& params) {
        const ov::PartialShape data_shape{1, 1, 2, 5};
        params.data = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfyx,
            std::vector<TypeParam>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
        params.updates = this->template AllocateTensor<TypeParam>(
            ov::PartialShape{1, 1, 1, 5}, format::bfyx,
            std::vector<TypeParam>{10, 20, 30, 40, 50});
        params.start = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {0});
        params.stop = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {1});
        params.step = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {1});
        params.axes = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {2});
        params.wanted_output = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfyx,
            std::vector<TypeParam>{10, 20, 30, 40, 50, 5, 6, 7, 8, 9});
    }

    // Test case: Spec Example 3 - multi-axis step=2 without explicit axes
    // data [1,1,3,5] = [[0..4],[5..9],[10..14]]
    // updates [1,1,2,2] = [[50,60],[70,80]]
    // start=[0,0,0,1], stop=[1,1,3,5], step=[1,1,2,2]
    // axis 2 (y): y=0,2  axis 3 (x): x=1,3
    // Expected: [[0,50,2,60,4],[5,6,7,8,9],[10,70,12,80,14]]
    template<typename TypeParam>
    void FillWithSpecExample3Data(SliceScatterTestParams& params) {
        const ov::PartialShape data_shape{1, 1, 3, 5};
        params.data = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfyx,
            std::vector<TypeParam>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14});
        params.updates = this->template AllocateTensor<TypeParam>(
            ov::PartialShape{1, 1, 2, 2}, format::bfyx,
            std::vector<TypeParam>{50, 60, 70, 80});
        params.start = this->template AllocateTensor<int64_t>(
            ov::PartialShape{4}, format::bfyx, {0, 0, 0, 1});
        params.stop = this->template AllocateTensor<int64_t>(
            ov::PartialShape{4}, format::bfyx, {1, 1, 3, 5});
        params.step = this->template AllocateTensor<int64_t>(
            ov::PartialShape{4}, format::bfyx, {1, 1, 2, 2});
        params.wanted_output = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfyx,
            std::vector<TypeParam>{0, 50, 2, 60, 4, 5, 6, 7, 8, 9, 10, 70, 12, 80, 14});
    }

    // Test case: 1D tensor
    // data [8] = [0,1,2,3,4,5,6,7]
    // updates [3] = [10,20,30]
    // start=[2], stop=[5], step=[1], axes=[0]
    // x[2:5] = [10,20,30]
    // Expected: [0,1,10,20,30,5,6,7]
    template<typename TypeParam>
    void FillWith1DData(SliceScatterTestParams& params) {
        const ov::PartialShape data_shape{8};
        params.data = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfyx, helpers::GenInput<TypeParam>(data_shape));
        params.updates = this->template AllocateTensor<TypeParam>(
            ov::PartialShape{3}, format::bfyx,
            std::vector<TypeParam>{10, 20, 30});
        params.start = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {2});
        params.stop = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {5});
        params.step = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {1});
        params.axes = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {0});
        params.wanted_output = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfyx,
            std::vector<TypeParam>{0, 1, 10, 20, 30, 5, 6, 7});
    }

    // Test case: 2D tensor
    // data [3,4] = GenInput(0..11) = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    // updates [3,2] = [100,101,102,103,104,105]
    // start=[1], stop=[3], step=[1], axes=[1]
    // x[1:3] on each row
    // Expected: [[0,100,101,3],[4,102,103,7],[8,104,105,11]]
    template<typename TypeParam>
    void FillWith2DData(SliceScatterTestParams& params) {
        const ov::PartialShape data_shape{3, 4};
        params.data = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfyx, helpers::GenInput<TypeParam>(data_shape));
        params.updates = this->template AllocateTensor<TypeParam>(
            ov::PartialShape{3, 2}, format::bfyx,
            std::vector<TypeParam>{100, 101, 102, 103, 104, 105});
        params.start = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {1});
        params.stop = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {3});
        params.step = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {1});
        params.axes = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {1});
        params.wanted_output = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfyx,
            std::vector<TypeParam>{0, 100, 101, 3, 4, 102, 103, 7, 8, 104, 105, 11});
    }

    // Test case: INT_MAX stop (slice to end of dimension)
    // data [1,1,1,10] = GenInput(0..9)
    // updates [1,1,1,4] = [50,51,52,53]
    // start=[6], stop=[INT_MAX], step=[1], axes=[3]
    // x[6:10] = [50,51,52,53]
    // Expected: [0,1,2,3,4,5,50,51,52,53]
    template<typename TypeParam>
    void FillWithIntMaxStopData(SliceScatterTestParams& params) {
        const ov::PartialShape data_shape{1, 1, 1, 10};
        params.data = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfyx, helpers::GenInput<TypeParam>(data_shape));
        params.updates = this->template AllocateTensor<TypeParam>(
            ov::PartialShape{1, 1, 1, 4}, format::bfyx,
            std::vector<TypeParam>{50, 51, 52, 53});
        params.start = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {6});
        params.stop = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {std::numeric_limits<int64_t>::max()});
        params.step = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {1});
        params.axes = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {3});
        params.wanted_output = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfyx,
            std::vector<TypeParam>{0, 1, 2, 3, 4, 5, 50, 51, 52, 53});
    }

    // Test case: negative step=-1 (full reverse on one axis)
    // data [1,1,1,6] = [0,1,2,3,4,5]
    // updates [1,1,1,3] = [10,11,12]
    // start=[5], stop=[2], step=[-1], axes=[3]
    // Backward: x=5,4,3 => positions 5,4,3
    // Expected: [0,1,2,12,11,10]
    template<typename TypeParam>
    void FillWithNegativeStepReverseData(SliceScatterTestParams& params) {
        const ov::PartialShape data_shape{1, 1, 1, 6};
        params.data = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfyx, helpers::GenInput<TypeParam>(data_shape));
        params.updates = this->template AllocateTensor<TypeParam>(
            ov::PartialShape{1, 1, 1, 3}, format::bfyx,
            std::vector<TypeParam>{10, 11, 12});
        params.start = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {5});
        params.stop = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {2});
        params.step = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {-1});
        params.axes = this->template AllocateTensor<int64_t>(
            ov::PartialShape{1}, format::bfyx, {3});
        params.wanted_output = this->template AllocateTensor<TypeParam>(
            data_shape, format::bfyx,
            std::vector<TypeParam>{0, 1, 2, 12, 11, 10});
    }

private:
    void SetParameterInput(const std::string& name, topology& topo,
                           const memory::ptr& data_ptr, bool is_dynamic) {
        if (is_dynamic) {
            auto dynamic_shape = data_ptr->get_layout();
            dynamic_shape.set_partial_shape(ov::PartialShape::dynamic(dynamic_shape.get_rank()));
            topo.add(input_layout(name, dynamic_shape));
        } else {
            topo.add(data(name, data_ptr));
        }
    }

    void RunTestCase(const SliceScatterTestParams& params) {
        topology topo;

        // data input - always input_layout
        auto data_layout = params.data->get_layout();
        if (params.is_data_dynamic) {
            data_layout.set_partial_shape(ov::PartialShape::dynamic(data_layout.get_rank()));
        }
        topo.add(input_layout("data", data_layout));

        // updates input - always input_layout
        auto updates_layout = params.updates->get_layout();
        if (params.is_updates_dynamic) {
            updates_layout.set_partial_shape(ov::PartialShape::dynamic(updates_layout.get_rank()));
        }
        topo.add(input_layout("updates", updates_layout));

        // start, stop, step: can be data (constant) or input_layout (dynamic)
        SetParameterInput("start", topo, params.start, params.is_start_dynamic);
        SetParameterInput("stop", topo, params.stop, params.is_stop_dynamic);
        SetParameterInput("step", topo, params.step, params.is_step_dynamic);

        if (params.axes) {
            SetParameterInput("axes", topo, params.axes, params.is_axes_dynamic);
        }

        std::vector<input_info> inputs{
            input_info("data"),
            input_info("updates"),
            input_info("start"),
            input_info("stop"),
            input_info("step")
        };
        if (params.axes) {
            inputs.push_back(input_info("axes"));
        }
        topo.add(slice_scatter("slice_scatter", inputs));

        ExecutionConfig config = get_test_default_config(engine_);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        cldnn::network::ptr network =
            get_network(engine_, topo, config, get_test_stream_ptr(), params.is_caching_test);

        network->set_input_data("data", params.data);
        network->set_input_data("updates", params.updates);

        if (params.is_start_dynamic)
            network->set_input_data("start", params.start);
        if (params.is_stop_dynamic)
            network->set_input_data("stop", params.stop);
        if (params.is_step_dynamic)
            network->set_input_data("step", params.step);
        if (params.axes && params.is_axes_dynamic)
            network->set_input_data("axes", params.axes);

        auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "slice_scatter");

        auto output = outputs.at("slice_scatter").get_memory();

        cldnn::mem_lock<T> output_ptr(output, get_test_stream());
        cldnn::mem_lock<T> wanted_output_ptr(params.wanted_output, get_test_stream());

        ASSERT_EQ(output->get_layout().get_shape(), params.wanted_output->get_layout().get_shape());
        ASSERT_EQ(output_ptr.size(), wanted_output_ptr.size());
        for (size_t i = 0; i < output_ptr.size(); ++i)
            ASSERT_TRUE(are_equal(wanted_output_ptr[i], output_ptr[i], 2e-3));
    }

    engine& engine_ = get_test_engine();
};

using testing::Types;
typedef Types<float, int32_t, int64_t> DataTypes;
TYPED_TEST_SUITE(SliceScatterTest, DataTypes);

// ===== Static shape tests =====

TYPED_TEST(SliceScatterTest, basic_bfyx) {
    SliceScatterTestParams params;
    this->template FillWithBasicBfyxData<TypeParam>(params);
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceScatterTest, basic_bfyx_caching) {
    SliceScatterTestParams params;
    this->template FillWithBasicBfyxData<TypeParam>(params);
    params.is_caching_test = true;
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceScatterTest, with_axes) {
    SliceScatterTestParams params;
    this->template FillWithAxesData<TypeParam>(params);
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceScatterTest, with_step) {
    SliceScatterTestParams params;
    this->template FillWithStepData<TypeParam>(params);
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceScatterTest, negative_indices) {
    SliceScatterTestParams params;
    this->template FillWithNegativeIndicesData<TypeParam>(params);
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceScatterTest, bfzyx) {
    SliceScatterTestParams params;
    this->template FillWithBfzyxData<TypeParam>(params);
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceScatterTest, negative_step) {
    SliceScatterTestParams params;
    this->template FillWithNegativeStepData<TypeParam>(params);
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceScatterTest, negative_step_reverse) {
    SliceScatterTestParams params;
    this->template FillWithNegativeStepReverseData<TypeParam>(params);
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceScatterTest, clamping) {
    SliceScatterTestParams params;
    this->template FillWithClampingData<TypeParam>(params);
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceScatterTest, int_max_stop) {
    SliceScatterTestParams params;
    this->template FillWithIntMaxStopData<TypeParam>(params);
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceScatterTest, negative_axes) {
    SliceScatterTestParams params;
    this->template FillWithNegativeAxesData<TypeParam>(params);
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceScatterTest, spec_example1) {
    SliceScatterTestParams params;
    this->template FillWithSpecExample1Data<TypeParam>(params);
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceScatterTest, spec_example3) {
    SliceScatterTestParams params;
    this->template FillWithSpecExample3Data<TypeParam>(params);
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceScatterTest, tensor_1d) {
    SliceScatterTestParams params;
    this->template FillWith1DData<TypeParam>(params);
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceScatterTest, tensor_2d) {
    SliceScatterTestParams params;
    this->template FillWith2DData<TypeParam>(params);
    this->RunAllTestCasesForParams(params);
}

// ===== Dynamic shape tests =====

TYPED_TEST(SliceScatterTest, basic_bfyx_data_dynamic) {
    SliceScatterTestParams params;
    this->template FillWithBasicBfyxData<TypeParam>(params);
    params.is_data_dynamic = true;
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceScatterTest, basic_bfyx_updates_dynamic) {
    SliceScatterTestParams params;
    this->template FillWithBasicBfyxData<TypeParam>(params);
    params.is_updates_dynamic = true;
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceScatterTest, basic_bfyx_start_dynamic) {
    SliceScatterTestParams params;
    this->template FillWithBasicBfyxData<TypeParam>(params);
    params.is_start_dynamic = true;
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceScatterTest, basic_bfyx_all_dynamic) {
    SliceScatterTestParams params;
    this->template FillWithBasicBfyxData<TypeParam>(params);
    params.is_data_dynamic = true;
    params.is_updates_dynamic = true;
    params.is_start_dynamic = true;
    params.is_stop_dynamic = true;
    params.is_step_dynamic = true;
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceScatterTest, basic_bfyx_all_dynamic_caching) {
    SliceScatterTestParams params;
    this->template FillWithBasicBfyxData<TypeParam>(params);
    params.is_data_dynamic = true;
    params.is_updates_dynamic = true;
    params.is_start_dynamic = true;
    params.is_stop_dynamic = true;
    params.is_step_dynamic = true;
    params.is_caching_test = true;
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceScatterTest, with_axes_all_dynamic) {
    SliceScatterTestParams params;
    this->template FillWithAxesData<TypeParam>(params);
    params.is_data_dynamic = true;
    params.is_updates_dynamic = true;
    params.is_start_dynamic = true;
    params.is_stop_dynamic = true;
    params.is_step_dynamic = true;
    params.is_axes_dynamic = true;
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceScatterTest, with_step_dynamic) {
    SliceScatterTestParams params;
    this->template FillWithStepData<TypeParam>(params);
    params.is_data_dynamic = true;
    params.is_step_dynamic = true;
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceScatterTest, negative_indices_all_dynamic) {
    SliceScatterTestParams params;
    this->template FillWithNegativeIndicesData<TypeParam>(params);
    params.is_data_dynamic = true;
    params.is_updates_dynamic = true;
    params.is_start_dynamic = true;
    params.is_stop_dynamic = true;
    params.is_step_dynamic = true;
    params.is_axes_dynamic = true;
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceScatterTest, bfzyx_all_dynamic) {
    SliceScatterTestParams params;
    this->template FillWithBfzyxData<TypeParam>(params);
    params.is_data_dynamic = true;
    params.is_updates_dynamic = true;
    params.is_start_dynamic = true;
    params.is_stop_dynamic = true;
    params.is_step_dynamic = true;
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceScatterTest, negative_step_all_dynamic) {
    SliceScatterTestParams params;
    this->template FillWithNegativeStepData<TypeParam>(params);
    params.is_data_dynamic = true;
    params.is_updates_dynamic = true;
    params.is_start_dynamic = true;
    params.is_stop_dynamic = true;
    params.is_step_dynamic = true;
    params.is_axes_dynamic = true;
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceScatterTest, clamping_all_dynamic) {
    SliceScatterTestParams params;
    this->template FillWithClampingData<TypeParam>(params);
    params.is_data_dynamic = true;
    params.is_updates_dynamic = true;
    params.is_start_dynamic = true;
    params.is_stop_dynamic = true;
    params.is_step_dynamic = true;
    params.is_axes_dynamic = true;
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceScatterTest, negative_axes_all_dynamic) {
    SliceScatterTestParams params;
    this->template FillWithNegativeAxesData<TypeParam>(params);
    params.is_data_dynamic = true;
    params.is_updates_dynamic = true;
    params.is_start_dynamic = true;
    params.is_stop_dynamic = true;
    params.is_step_dynamic = true;
    params.is_axes_dynamic = true;
    this->RunAllTestCasesForParams(params);
}

TYPED_TEST(SliceScatterTest, tensor_1d_all_dynamic) {
    SliceScatterTestParams params;
    this->template FillWith1DData<TypeParam>(params);
    params.is_data_dynamic = true;
    params.is_updates_dynamic = true;
    params.is_start_dynamic = true;
    params.is_stop_dynamic = true;
    params.is_step_dynamic = true;
    params.is_axes_dynamic = true;
    this->RunAllTestCasesForParams(params);
}

} // anonymous namespace

// ===== Performance benchmark (large tensors, opt-eligible: step=1, X>=8) =====
namespace {

class SliceScatterPerfTest : public ::testing::Test {
protected:
    engine& engine_ = get_test_engine();

    // Benchmark: run the network 'iters' times and return average execution time in microseconds
    double BenchmarkSliceScatter(const ov::PartialShape& data_shape,
                                 const ov::PartialShape& updates_shape,
                                 const std::vector<int64_t>& start_vals,
                                 const std::vector<int64_t>& stop_vals,
                                 const std::vector<int64_t>& step_vals,
                                 int warmup_iters, int measure_iters) {
        // Allocate data
        auto data_size = ov::shape_size(data_shape.get_shape());
        auto updates_size = ov::shape_size(updates_shape.get_shape());
        std::vector<float> data_vals(data_size);
        std::iota(data_vals.begin(), data_vals.end(), 0.0f);
        std::vector<float> upd_vals(updates_size, 1.0f);

        layout data_lo{data_shape, data_types::f32, format::bfyx};
        layout updates_lo{updates_shape, data_types::f32, format::bfyx};
        auto data_mem = engine_.allocate_memory(data_lo);
        auto updates_mem = engine_.allocate_memory(updates_lo);
        set_values<float>(data_mem, data_vals);
        set_values<float>(updates_mem, upd_vals);

        ov::PartialShape idx_shape{static_cast<int64_t>(start_vals.size())};
        layout idx_lo{idx_shape, data_types::i64, format::bfyx};
        auto start_mem = engine_.allocate_memory(idx_lo);
        auto stop_mem = engine_.allocate_memory(idx_lo);
        auto step_mem = engine_.allocate_memory(idx_lo);
        set_values<int64_t>(start_mem, start_vals);
        set_values<int64_t>(stop_mem, stop_vals);
        set_values<int64_t>(step_mem, step_vals);

        // Build topology
        topology topo;
        topo.add(input_layout("data", data_lo));
        topo.add(input_layout("updates", updates_lo));
        topo.add(data("start", start_mem));
        topo.add(data("stop", stop_mem));
        topo.add(data("step", step_mem));
        topo.add(slice_scatter("slice_scatter",
                               {input_info("data"), input_info("updates"),
                                input_info("start"), input_info("stop"), input_info("step")}));

        ExecutionConfig config = get_test_default_config(engine_);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        auto network = get_network(engine_, topo, config, get_test_stream_ptr(), false);

        // Warmup
        for (int i = 0; i < warmup_iters; ++i) {
            network->set_input_data("data", data_mem);
            network->set_input_data("updates", updates_mem);
            auto outputs = network->execute();
            // Force sync
            auto out = outputs.at("slice_scatter").get_memory();
            cldnn::mem_lock<float> lock(out, get_test_stream());
            (void)lock[0];
        }

        // Measure
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < measure_iters; ++i) {
            network->set_input_data("data", data_mem);
            network->set_input_data("updates", updates_mem);
            auto outputs = network->execute();
            auto out = outputs.at("slice_scatter").get_memory();
            cldnn::mem_lock<float> lock(out, get_test_stream());
            (void)lock[0];
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double total_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        return total_us / measure_iters;
    }
};

TEST_F(SliceScatterPerfTest, large_tensor_perf) {
    // Large tensor: data [4, 16, 64, 256], updates [4, 16, 32, 256]
    // step=1 on all axes => opt kernel eligible (X=256 >> VEC_SIZE=8)
    const ov::PartialShape data_shape{4, 16, 64, 256};
    const ov::PartialShape updates_shape{4, 16, 32, 256};
    std::vector<int64_t> start_vals{0, 0, 0, 0};
    std::vector<int64_t> stop_vals{4, 16, 32, 256};
    std::vector<int64_t> step_vals{1, 1, 1, 1};

    const int warmup = 10;
    const int iters = 50;

    double avg_us = BenchmarkSliceScatter(data_shape, updates_shape,
                                          start_vals, stop_vals, step_vals,
                                          warmup, iters);
    std::cout << "[PERF] SliceScatter large tensor (4x64x128x256, step=1): "
              << avg_us << " us/iter (" << iters << " iters)" << std::endl;
    ASSERT_GT(avg_us, 0.0);
}

TEST_F(SliceScatterPerfTest, medium_tensor_perf) {
    // Medium tensor: data [8, 32, 64, 64], updates [8, 32, 32, 64]
    const ov::PartialShape data_shape{8, 32, 64, 64};
    const ov::PartialShape updates_shape{8, 32, 32, 64};
    std::vector<int64_t> start_vals{0, 0, 0, 0};
    std::vector<int64_t> stop_vals{8, 32, 32, 64};
    std::vector<int64_t> step_vals{1, 1, 1, 1};

    const int warmup = 10;
    const int iters = 100;

    double avg_us = BenchmarkSliceScatter(data_shape, updates_shape,
                                          start_vals, stop_vals, step_vals,
                                          warmup, iters);
    std::cout << "[PERF] SliceScatter medium tensor (8x32x64x64, step=1): "
              << avg_us << " us/iter (" << iters << " iters)" << std::endl;
    ASSERT_GT(avg_us, 0.0);
}

} // anonymous namespace
