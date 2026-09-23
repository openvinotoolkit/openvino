// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <atomic>
#include <thread>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/node_builders/activation.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "transformations/utils/utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/subgraph_builders/split_multi_conv_concat.hpp"
#include "common_test_utils/subgraph_builders/read_concat_split_assign.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"
#include "openvino/runtime/intel_gpu/ocl/ocl_wrapper.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"

namespace {
typedef std::tuple<
        ov::element::Type,   // Input/Output type
        ov::Shape,           // Input Shape
        std::string> newtworkParams;

class InferRequestIOPrecision : public testing::WithParamInterface<newtworkParams>,
                                virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<newtworkParams> &obj);

protected:
    void SetUp() override;
};

std::string InferRequestIOPrecision::getTestCaseName(const testing::TestParamInfo<newtworkParams> &obj) {
    const auto& [model_type, shape, targetDevice] = obj.param;

    std::ostringstream result;
    const char separator = '_';
    result << "netPRC=" << model_type.get_type_name() << separator;
    result << "trgDev=" << targetDevice;
    return result.str();
}

void InferRequestIOPrecision::SetUp() {
    const auto& [model_type, shape, _targetDevice] = GetParam();
    targetDevice = _targetDevice;

    float clamp_min = model_type.is_signed() ? -5.f : 0.0f;
    float clamp_max = 5.0f;

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape(shape))};
    params[0]->set_friendly_name("Input");

    auto activation = ov::test::utils::make_activation(params[0],
                                                       model_type,
                                                       ov::test::utils::ActivationTypes::Clamp,
                                                       {},
                                                       {clamp_min, clamp_max});

    function = std::make_shared<ov::Model>(ov::OutputVector{activation}, params);
}

TEST_P(InferRequestIOPrecision, Inference) {
    run();
}

const std::vector<ov::element::Type> input_types = {
        ov::element::i16,
        ov::element::u16,
        ov::element::f32,
        ov::element::f16,
        ov::element::u8,
        ov::element::i8,
        ov::element::i32,
        ov::element::u32,
        ov::element::u64,
        ov::element::i64,
        // Interpreter backend doesn't implement evaluate method for OP
        // ov::element::f64,
};

INSTANTIATE_TEST_SUITE_P(smoke_GPU_BehaviorTests, InferRequestIOPrecision,
                         ::testing::Combine(
                                 ::testing::ValuesIn(input_types),
                                 ::testing::Values(ov::Shape{1, 50}),
                                 ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         InferRequestIOPrecision::getTestCaseName);

static std::shared_ptr<ov::Model> makeStaticInputModel(ov::Shape& shape_out) {
    const ov::Shape shape{1, 64};
    shape_out = shape;
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    param->set_friendly_name("static_input");
    param->output(0).get_tensor().set_names({"static_input"});
    auto relu = std::make_shared<ov::op::v0::Relu>(param);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(relu)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{param});
}

// Caller-owned USM-host output sharing needs an integrated GPU whose driver actually exposes USM;
// RemoteContextImpl rejects USM_HOST allocation when unified shared memory is unsupported/disabled.
static bool gpu_supports_usm_host_output_sharing(ov::Core& core) {
    if (core.get_property(ov::test::utils::DEVICE_GPU, ov::device::type) != ov::device::Type::INTEGRATED)
        return false;
    const auto caps = core.get_property(ov::test::utils::DEVICE_GPU, ov::device::capabilities);
    return std::find(caps.begin(), caps.end(), ov::intel_gpu::capability::USM_MEMORY) != caps.end();
}

// Static input must be inferable when set_tensor() is never called.
TEST(TensorTest, smoke_lazyAllocStaticInputInferWithoutSetTensor) {
    ov::Shape shape;
    auto model = makeStaticInputModel(shape);

    auto core = ov::Core();
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU);
    auto request = compiled_model.create_infer_request();

    OV_ASSERT_NO_THROW(request.infer());

    ov::Tensor in;
    OV_ASSERT_NO_THROW(in = request.get_input_tensor(0));
    ASSERT_EQ(in.get_shape(), shape);
    ASSERT_NE(in.data(), nullptr);
    OV_ASSERT_NO_THROW(request.get_output_tensor(0));
}

// get_tensor() before infer() must also materialize the lazy slot.
TEST(TensorTest, smoke_lazyAllocStaticInputGetTensorBeforeInfer) {
    ov::Shape shape;
    auto model = makeStaticInputModel(shape);

    auto core = ov::Core();
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU);
    auto request = compiled_model.create_infer_request();

    ov::Tensor in;
    OV_ASSERT_NO_THROW(in = request.get_input_tensor(0));
    ASSERT_EQ(in.get_shape(), shape);
    ASSERT_NE(in.data(), nullptr);

    OV_ASSERT_NO_THROW(request.infer());
}

TEST(TensorTest, smoke_lazyAllocStaticOutputReusesUserTensor) {
    ov::Shape shape;
    auto model = makeStaticInputModel(shape);

    auto core = ov::Core();
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU);
    auto request = compiled_model.create_infer_request();

    ov::Tensor input(ov::element::f32, shape);
    ov::Tensor output(ov::element::f32, shape);
    request.set_input_tensor(input);
    request.set_output_tensor(output);

    for (float value : {1.0f, 2.0f}) {
        std::fill_n(input.data<float>(), input.get_size(), value);

        OV_ASSERT_NO_THROW(request.infer());

        auto actual = request.get_output_tensor();
        ASSERT_EQ(actual.data(), output.data());
        for (size_t i = 0; i < actual.get_size(); ++i) {
            ASSERT_FLOAT_EQ(actual.data<const float>()[i], value);
        }
    }
}

// Dynamic output bound to a caller-owned iGPU USM-host buffer produces correct values.
TEST(TensorTest, smoke_dynamicOutputCallerOwnedUsmHost) {
    auto core = ov::Core();
    if (!gpu_supports_usm_host_output_sharing(core)) {
        GTEST_SKIP() << "Caller-owned USM-host output sharing requires an iGPU with USM support";
    }

    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 4});
    auto relu = std::make_shared<ov::op::v0::Relu>(input);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{relu}, ov::ParameterVector{input});

    auto remote_context = core.get_default_context(ov::test::utils::DEVICE_GPU);
    auto compiled_model = core.compile_model(model, remote_context);

    const ov::Shape shape{2, 4};
    ov::Tensor input_tensor(ov::element::f32, shape);
    const std::vector<float> input_values{-4.0f, -3.0f, -2.0f, -1.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    std::copy(input_values.begin(), input_values.end(), input_tensor.data<float>());

    auto gpu_context = remote_context.as<ov::intel_gpu::ocl::ClContext>();
    auto usm_allocation = gpu_context.create_usm_host_tensor(ov::element::f32, shape);
    ov::Tensor output_tensor(ov::element::f32, shape, usm_allocation.get());
    ASSERT_FALSE(output_tensor.is<ov::intel_gpu::ocl::USMTensor>());
    std::fill_n(output_tensor.data<float>(), output_tensor.get_size(), -1.0f);

    auto request = compiled_model.create_infer_request();
    request.set_input_tensor(input_tensor);
    request.set_output_tensor(output_tensor);
    OV_ASSERT_NO_THROW(request.infer());

    const std::vector<float> expected{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    auto actual = request.get_output_tensor();
    // This asserts the caller-pointer contract and correctness, not that zero-copy actually happened
    // (data() stays the caller pointer even on the copy path). The graph-shares-caller-allocation
    // assertion lives in unit/dynamic_execution/zero_copy_output_test.cpp, which can see graph memory.
    ASSERT_EQ(actual.data(), usm_allocation.get());
    ASSERT_EQ(actual.get_size(), expected.size());
    for (size_t i = 0; i < actual.get_size(); ++i) {
        ASSERT_FLOAT_EQ(actual.data<const float>()[i], expected[i]);
    }
}

static std::shared_ptr<ov::Model> makeDynamicReluModel() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 4});
    auto relu = std::make_shared<ov::op::v0::Relu>(input);
    return std::make_shared<ov::Model>(ov::OutputVector{relu}, ov::ParameterVector{input});
}

static std::shared_ptr<ov::Model> makeDynamicMatMulModel(const size_t k, std::vector<float>& weights_data) {
    weights_data.resize(k * k);
    for (size_t i = 0; i < weights_data.size(); ++i) {
        weights_data[i] = static_cast<float>((i % 7) + 1) * 0.01f;
    }

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, static_cast<int64_t>(k)});
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{k, k}, weights_data);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(param, weights, false, false);
    return std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{param});
}

static std::vector<float> reference_matmul(const ov::Shape& shape,
                                           const size_t k,
                                           const std::vector<float>& input_values,
                                           const std::vector<float>& weights_data) {
    std::vector<float> expected(ov::shape_size(shape), 0.0f);
    for (size_t r = 0; r < shape[0]; ++r) {
        for (size_t c = 0; c < k; ++c) {
            float acc = 0.0f;
            for (size_t col = 0; col < k; ++col) {
                acc += input_values[r * k + col] * weights_data[col * k + c];
            }
            expected[r * k + c] = acc;
        }
    }
    return expected;
}

static int64_t gpu_mem_in_use(const ov::Core& core) {
    int64_t total = 0;
    for (const auto& [type, bytes] : core.get_property(ov::test::utils::DEVICE_GPU, ov::intel_gpu::memory_statistics)) {
        total += static_cast<int64_t>(bytes);
    }
    return total;
}

static int64_t f32_bytes(const ov::Shape& shape) {
    return static_cast<int64_t>(ov::shape_size(shape) * sizeof(float));
}

// Imported caller USM isn't engine-tracked, so a copy fallback shows up as >= one extra output buffer.
// Half an output buffer of margin absorbs run-to-run variance but still catches a full extra buffer.
static void expect_extra_output_buffer(int64_t growth, int64_t baseline_growth, int64_t output_bytes) {
    EXPECT_GE(growth - baseline_growth, output_bytes - output_bytes / 2)
        << "growth=" << growth << " B, baseline=" << baseline_growth << " B, output=" << output_bytes
        << " B: expected a plugin-owned output buffer (copy fallback)";
}

static void expect_no_extra_output_buffer(int64_t growth, int64_t baseline_growth, int64_t output_bytes) {
    EXPECT_LT(growth - baseline_growth, output_bytes / 2)
        << "growth=" << growth << " B, baseline=" << baseline_growth << " B, output=" << output_bytes
        << " B: caller USM-host output likely fell back to a plugin-owned copy";
}

// Small integers stay exact through the default f16 inference precision.
static float relu_test_input(size_t i) {
    return static_cast<float>(i % 17) - 8.0f;
}

static void fill_relu_test_input(ov::Tensor& tensor) {
    auto* data = tensor.data<float>();
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        data[i] = relu_test_input(i);
    }
}

static bool matches_relu_of_test_input(const ov::Tensor& tensor) {
    const auto* data = tensor.data<const float>();
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        const float in = relu_test_input(i);
        if (data[i] != (in > 0.0f ? in : 0.0f)) {
            return false;
        }
    }
    return true;
}

// Experimental: zero-copy must allocate ~one output buffer less than the same run with an ordinary
// host output (copy fallback).
TEST(TensorTest, smoke_dynamicOutputCallerOwnedUsmHostAllocatesLessThanCopyFallback) {
    {
        auto core = ov::Core();
        if (!gpu_supports_usm_host_output_sharing(core)) {
            GTEST_SKIP() << "Caller-owned USM-host output sharing requires an iGPU with USM support";
        }
    }

    // makeDynamicReluModel is [?, 4]; 262144 rows makes a 4 MB f32 output so it dominates noise.
    const ov::Shape shape{262144, 4};
    constexpr size_t kIterations = 3;

    // Each variant uses its own Core so graph-held buffers from the other variant can't leak in.
    auto measure = [&](bool usm_output) -> int64_t {
        auto core = ov::Core();
        auto remote_context = core.get_default_context(ov::test::utils::DEVICE_GPU);
        auto compiled_model = core.compile_model(makeDynamicReluModel(), remote_context);
        auto gpu_context = remote_context.as<ov::intel_gpu::ocl::ClContext>();
        auto request = compiled_model.create_infer_request();

        // Allocated in both variants so the caller buffer's own tracked bytes don't skew the delta.
        auto usm_allocation = gpu_context.create_usm_host_tensor(ov::element::f32, shape);
        std::vector<float> host_output(ov::shape_size(shape));
        ov::Tensor input_tensor(ov::element::f32, shape);
        fill_relu_test_input(input_tensor);
        void* output_ptr = usm_output ? usm_allocation.get() : static_cast<void*>(host_output.data());
        request.set_input_tensor(input_tensor);
        request.set_output_tensor(ov::Tensor(ov::element::f32, shape, output_ptr));

        const int64_t before = gpu_mem_in_use(core);
        for (size_t iter = 0; iter < kIterations; ++iter) {
            request.infer();
        }
        const int64_t delta = gpu_mem_in_use(core) - before;

        auto actual = request.get_output_tensor();
        EXPECT_EQ(actual.data(), output_ptr);
        EXPECT_TRUE(matches_relu_of_test_input(actual)) << "usm_output=" << usm_output;
        return delta;
    };

    const int64_t zero_copy_delta = measure(true);
    const int64_t copy_delta = measure(false);
    expect_extra_output_buffer(copy_delta, zero_copy_delta, f32_bytes(shape));
}

// A caller buffer sized for the max shape stays bound and correct across smaller/larger runtime shapes.
TEST(TensorTest, smoke_dynamicOutputCallerOwnedUsmHostGrowthWithinCapacity) {
    {
        auto core = ov::Core();
        if (!gpu_supports_usm_host_output_sharing(core)) {
            GTEST_SKIP() << "Caller-owned USM-host output sharing requires an iGPU with USM support";
        }
    }

    // Rows are scaled so the max output (8 MB) dominates allocation noise.
    constexpr size_t kRowScale = 65536;
    const ov::Shape max_shape{8 * kRowScale, 4};

    // Same run with a caller USM-host output vs an ordinary host output (copy fallback), each in its own Core.
    auto measure = [&](bool usm_output) -> int64_t {
        auto core = ov::Core();
        auto compiled_model = core.compile_model(makeDynamicReluModel(), core.get_default_context(ov::test::utils::DEVICE_GPU));
        auto gpu_context = compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();
        auto request = compiled_model.create_infer_request();

        auto usm_allocation = gpu_context.create_usm_host_tensor(ov::element::f32, max_shape);
        std::vector<float> host_output(ov::shape_size(max_shape));
        void* output_ptr = usm_output ? usm_allocation.get() : static_cast<void*>(host_output.data());

        // Bind the max-capacity caller buffer exactly once; the loop varies only the input shape.
        request.set_output_tensor(ov::Tensor(ov::element::f32, max_shape, output_ptr));

        const int64_t before = gpu_mem_in_use(core);
        for (const size_t rows : {size_t{2}, size_t{8}, size_t{4}}) {
            const ov::Shape shape{rows * kRowScale, 4};
            // The first bind (pre-shrink) fixes capacity at max_shape and same-pointer rebinds are
            // no-ops (is_the_same_buffer), so the caller buffer stays zero-copy across all shapes.

            ov::Tensor input_tensor(ov::element::f32, shape);
            fill_relu_test_input(input_tensor);
            request.set_input_tensor(input_tensor);
            request.infer();

            auto actual = request.get_output_tensor();
            EXPECT_EQ(actual.data(), output_ptr);
            EXPECT_EQ(actual.get_shape(), shape);
            EXPECT_TRUE(matches_relu_of_test_input(actual)) << "rows=" << shape[0] << " usm_output=" << usm_output;
        }
        return gpu_mem_in_use(core) - before;
    };

    const int64_t zero_copy_growth = measure(true);
    const int64_t copy_growth = measure(false);
    expect_extra_output_buffer(copy_growth, zero_copy_growth, f32_bytes(max_shape));
}

// Runtime output exceeding the caller buffer fails safely (throws) without corrupting caller memory.
TEST(TensorTest, smoke_dynamicOutputCallerOwnedUsmHostGrowthBeyondCapacityIsSafe) {
    auto core = ov::Core();
    if (!gpu_supports_usm_host_output_sharing(core)) {
        GTEST_SKIP() << "Caller-owned USM-host output sharing requires an iGPU with USM support";
    }

    auto compiled_model = core.compile_model(makeDynamicReluModel(), core.get_default_context(ov::test::utils::DEVICE_GPU));
    auto gpu_context = compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();
    auto request = compiled_model.create_infer_request();

    const ov::Shape small_shape{2, 4};
    auto usm_allocation = gpu_context.create_usm_host_tensor(ov::element::f32, small_shape);
    ov::Tensor output_tensor(ov::element::f32, small_shape, usm_allocation.get());
    request.set_output_tensor(output_tensor);

    ov::Tensor small_input(ov::element::f32, small_shape);
    std::fill_n(small_input.data<float>(), small_input.get_size(), 1.0f);
    request.set_input_tensor(small_input);
    OV_ASSERT_NO_THROW(request.infer());
    ASSERT_EQ(request.get_output_tensor().data(), usm_allocation.get());

    // Sentinel detects any out-of-bounds write into the caller buffer during the oversized run.
    constexpr float sentinel = -17.0f;
    std::fill_n(static_cast<float*>(usm_allocation.get()), ov::shape_size(small_shape), sentinel);

    ov::Tensor large_input(ov::element::f32, ov::Shape{8, 4});
    std::fill_n(large_input.data<float>(), large_input.get_size(), 1.0f);
    request.set_input_tensor(large_input);
    ASSERT_ANY_THROW(request.infer());

    const auto* caller_data = static_cast<const float*>(usm_allocation.get());
    for (size_t i = 0; i < ov::shape_size(small_shape); ++i) {
        ASSERT_FLOAT_EQ(caller_data[i], sentinel);
    }
}

// Rebinding to a different caller USM-host allocation writes the new buffer, leaves the old untouched.
TEST(TensorTest, smoke_dynamicOutputCallerOwnedUsmHostRebindsAllocation) {
    auto core = ov::Core();
    if (!gpu_supports_usm_host_output_sharing(core)) {
        GTEST_SKIP() << "Caller-owned USM-host output sharing requires an iGPU with USM support";
    }

    auto compiled_model = core.compile_model(makeDynamicReluModel(), core.get_default_context(ov::test::utils::DEVICE_GPU));
    auto gpu_context = compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();
    auto request = compiled_model.create_infer_request();
    // Large enough that an extra plugin-owned output buffer stands out in tracked memory.
    const ov::Shape shape{262144, 4};

    auto first_allocation = gpu_context.create_usm_host_tensor(ov::element::f32, shape);
    auto second_allocation = gpu_context.create_usm_host_tensor(ov::element::f32, shape);
    ov::Tensor input_tensor(ov::element::f32, shape);

    std::fill_n(input_tensor.data<float>(), input_tensor.get_size(), 1.0f);
    request.set_input_tensor(input_tensor);
    request.set_output_tensor(ov::Tensor(ov::element::f32, shape, first_allocation.get()));
    OV_ASSERT_NO_THROW(request.infer());

    constexpr float sentinel = -23.0f;
    std::fill_n(static_cast<float*>(first_allocation.get()), ov::shape_size(shape), sentinel);
    std::fill_n(input_tensor.data<float>(), input_tensor.get_size(), 2.0f);
    request.set_output_tensor(ov::Tensor(ov::element::f32, shape, second_allocation.get()));
    const int64_t before_rebind = gpu_mem_in_use(core);
    OV_ASSERT_NO_THROW(request.infer());
    expect_no_extra_output_buffer(gpu_mem_in_use(core) - before_rebind, 0, f32_bytes(shape));

    auto actual = request.get_output_tensor();
    ASSERT_EQ(actual.data(), second_allocation.get());
    const auto* first_data = static_cast<const float*>(first_allocation.get());
    const auto* second_data = static_cast<const float*>(second_allocation.get());
    for (size_t i = 0; i < ov::shape_size(shape); ++i) {
        ASSERT_FLOAT_EQ(first_data[i], sentinel);
        ASSERT_FLOAT_EQ(second_data[i], 2.0f);
    }
}

// Switching from a bound caller USM-host buffer to an ineligible ordinary host tensor copies correctly.
TEST(TensorTest, smoke_dynamicOutputSwitchesFromUsmHostToCopyFallback) {
    auto core = ov::Core();
    if (!gpu_supports_usm_host_output_sharing(core)) {
        GTEST_SKIP() << "Caller-owned USM-host output sharing requires an iGPU with USM support";
    }

    auto compiled_model = core.compile_model(makeDynamicReluModel(), core.get_default_context(ov::test::utils::DEVICE_GPU));
    auto gpu_context = compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();
    auto request = compiled_model.create_infer_request();
    // Large enough that the plugin-owned copy buffer stands out in tracked memory.
    const ov::Shape shape{262144, 4};

    auto usm_allocation = gpu_context.create_usm_host_tensor(ov::element::f32, shape);
    ov::Tensor input_tensor(ov::element::f32, shape);
    std::fill_n(input_tensor.data<float>(), input_tensor.get_size(), 1.0f);
    request.set_input_tensor(input_tensor);
    request.set_output_tensor(ov::Tensor(ov::element::f32, shape, usm_allocation.get()));
    OV_ASSERT_NO_THROW(request.infer());

    constexpr float sentinel = -31.0f;
    std::fill_n(static_cast<float*>(usm_allocation.get()), ov::shape_size(shape), sentinel);
    std::vector<float> host_output(ov::shape_size(shape), sentinel);
    std::fill_n(input_tensor.data<float>(), input_tensor.get_size(), 3.0f);
    request.set_output_tensor(ov::Tensor(ov::element::f32, shape, host_output.data()));
    const int64_t before_switch = gpu_mem_in_use(core);
    OV_ASSERT_NO_THROW(request.infer());
    expect_extra_output_buffer(gpu_mem_in_use(core) - before_switch, 0, f32_bytes(shape));

    auto actual = request.get_output_tensor();
    ASSERT_EQ(actual.data(), host_output.data());
    const auto* usm_data = static_cast<const float*>(usm_allocation.get());
    for (size_t i = 0; i < ov::shape_size(shape); ++i) {
        ASSERT_FLOAT_EQ(usm_data[i], sentinel);
        ASSERT_FLOAT_EQ(host_output[i], 3.0f);
    }
}

// When the same caller USM-host buffer is supplied as both a dynamic output and an input,
// the zero-copy binding must be rejected: a tiled MatMul would otherwise overwrite the input
// buffer before it is fully read, silently corrupting results. The plugin must fall back to an
// internal output plus copy so the aliased input stays intact until the kernel finishes reading.
TEST(TensorTest, smoke_dynamicOutputCallerOwnedUsmHostInputOutputAliasIsSafe) {
    auto core = ov::Core();
    if (!gpu_supports_usm_host_output_sharing(core)) {
        GTEST_SKIP() << "Caller-owned USM-host output sharing requires an iGPU with USM support";
    }

    // K must be large enough to force tiled MatMul execution so reads/writes to the same
    // buffer interleave; square weights let the output be fed back as input.
    constexpr size_t K = 1024;
    const ov::Shape shape{2, K};

    // Known weights so the reference can be computed by hand.
    std::vector<float> weights_data;
    auto model = makeDynamicMatMulModel(K, weights_data);

    // f32 inference precision keeps the aliased buffer wired straight into MatMul: without it the
    // default f16 path inserts Convert/reorder nodes that stage input and output through separate
    // internal buffers, hiding the in-place aliasing this test must exercise.
    auto compiled_model = core.compile_model(model, core.get_default_context(ov::test::utils::DEVICE_GPU),
                                             ov::hint::inference_precision(ov::element::f32));
    auto gpu_context = compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();
    auto request = compiled_model.create_infer_request();

    // Single USM-host allocation used as BOTH input and output (the aliasing pattern).
    auto usm_allocation = gpu_context.create_usm_host_tensor(ov::element::f32, shape);
    auto* buffer = static_cast<float*>(usm_allocation.get());
    std::vector<float> input_values(ov::shape_size(shape));
    for (size_t i = 0; i < input_values.size(); ++i) {
        input_values[i] = static_cast<float>((i % 5)) - 2.0f;
    }

    const auto expected = reference_matmul(shape, K, input_values, weights_data);

    ov::Tensor aliased(ov::element::f32, shape, usm_allocation.get());
    request.set_input_tensor(aliased);
    request.set_output_tensor(aliased);

    // In-place aliasing corruption is a nondeterministic GPU data race: with the fix every run is
    // correct, without it a run is expected to diverge. Repeat so a regression is caught reliably.
    constexpr int kIterations = 16;
    for (int iter = 0; iter < kIterations; ++iter) {
        std::copy(input_values.begin(), input_values.end(), buffer);  // a prior corrupted run may have overwritten the input

        OV_ASSERT_NO_THROW(request.infer());

        auto actual = request.get_output_tensor();
        ASSERT_EQ(actual.get_size(), expected.size());
        const auto* actual_data = actual.data<const float>();
        for (size_t i = 0; i < actual.get_size(); ++i) {
            ASSERT_NEAR(actual_data[i], expected[i], 1e-2f)
                << "aliased input/output corrupted at element " << i << " on iteration " << iter;
        }
    }
}

TEST(TensorTest, smoke_dynamicOutputCallerOwnedUsmHostRemoteInputAliasIsSafe) {
    auto core = ov::Core();
    if (!gpu_supports_usm_host_output_sharing(core)) {
        GTEST_SKIP() << "Caller-owned USM-host output sharing requires an iGPU with USM support";
    }

    constexpr size_t K = 1024;
    const ov::Shape shape{2, K};
    std::vector<float> weights_data;
    auto model = makeDynamicMatMulModel(K, weights_data);
    auto compiled_model = core.compile_model(model,
                                             core.get_default_context(ov::test::utils::DEVICE_GPU),
                                             ov::hint::inference_precision(ov::element::f32));
    auto gpu_context = compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();
    auto request = compiled_model.create_infer_request();

    auto remote_input = gpu_context.create_usm_host_tensor(ov::element::f32, shape);
    auto* buffer = static_cast<float*>(remote_input.get());
    std::vector<float> input_values(ov::shape_size(shape));
    for (size_t i = 0; i < input_values.size(); ++i) {
        input_values[i] = static_cast<float>((i % 5)) - 2.0f;
    }
    const auto expected = reference_matmul(shape, K, input_values, weights_data);

    ov::Tensor output_wrapper(ov::element::f32, shape, remote_input.get());
    request.set_input_tensor(remote_input);
    request.set_output_tensor(output_wrapper);

    constexpr int kIterations = 16;
    for (int iter = 0; iter < kIterations; ++iter) {
        std::copy(input_values.begin(), input_values.end(), buffer);

        OV_ASSERT_NO_THROW(request.infer());

        auto actual = request.get_output_tensor();
        ASSERT_EQ(actual.get_size(), expected.size());
        const auto* actual_data = actual.data<const float>();
        for (size_t i = 0; i < actual.get_size(); ++i) {
            ASSERT_NEAR(actual_data[i], expected[i], 1e-2f)
                << "remote aliased input/output corrupted at element " << i << " on iteration " << iter;
        }
    }
}

// The same remote (USM-host) tensor is set as BOTH dynamic input and output. Since a remote output
// is normally bound zero-copy, the plugin must instead copy out so a tiled MatMul can't corrupt the input.
TEST(TensorTest, smoke_dynamicOutputCallerOwnedUsmHostRemoteOutputAliasIsSafe) {
    auto core = ov::Core();
    if (!gpu_supports_usm_host_output_sharing(core)) {
        GTEST_SKIP() << "Caller-owned USM-host output sharing requires an iGPU with USM support";
    }

    constexpr size_t K = 1024;
    const ov::Shape shape{2, K};
    std::vector<float> weights_data;
    auto model = makeDynamicMatMulModel(K, weights_data);
    auto compiled_model = core.compile_model(model,
                                             core.get_default_context(ov::test::utils::DEVICE_GPU),
                                             ov::hint::inference_precision(ov::element::f32));
    auto gpu_context = compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();
    auto request = compiled_model.create_infer_request();

    // A single remote tensor used as both input and output (the remote aliasing pattern).
    auto remote_aliased = gpu_context.create_usm_host_tensor(ov::element::f32, shape);
    auto* buffer = static_cast<float*>(remote_aliased.get());
    std::vector<float> input_values(ov::shape_size(shape));
    for (size_t i = 0; i < input_values.size(); ++i) {
        input_values[i] = static_cast<float>((i % 5)) - 2.0f;
    }
    const auto expected = reference_matmul(shape, K, input_values, weights_data);

    request.set_input_tensor(remote_aliased);
    request.set_output_tensor(remote_aliased);

    constexpr int kIterations = 16;
    for (int iter = 0; iter < kIterations; ++iter) {
        std::copy(input_values.begin(), input_values.end(), buffer);

        OV_ASSERT_NO_THROW(request.infer());

        // Output was copied back into the shared remote buffer; read it directly.
        for (size_t i = 0; i < expected.size(); ++i) {
            ASSERT_NEAR(buffer[i], expected[i], 1e-2f)
                << "remote aliased output corrupted at element " << i << " on iteration " << iter;
        }
    }
}

// A remote output that was bound zero-copy in an earlier non-aliased inference and only later
// aliases an input must drop the stale binding, otherwise set_output_memory() rebinds the aliased buffer.
TEST(TensorTest, smoke_dynamicOutputCallerOwnedUsmHostRemoteOutputBecomesAliasedIsSafe) {
    auto core = ov::Core();
    if (!gpu_supports_usm_host_output_sharing(core)) {
        GTEST_SKIP() << "Caller-owned USM-host output sharing requires an iGPU with USM support";
    }

    constexpr size_t K = 1024;
    const ov::Shape shape{2, K};
    std::vector<float> weights_data;
    auto model = makeDynamicMatMulModel(K, weights_data);
    auto compiled_model = core.compile_model(model,
                                             core.get_default_context(ov::test::utils::DEVICE_GPU),
                                             ov::hint::inference_precision(ov::element::f32));
    auto gpu_context = compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();
    auto request = compiled_model.create_infer_request();

    auto remote_output = gpu_context.create_usm_host_tensor(ov::element::f32, shape);
    auto* out_buffer = static_cast<float*>(remote_output.get());
    std::vector<float> input_values(ov::shape_size(shape));
    for (size_t i = 0; i < input_values.size(); ++i) {
        input_values[i] = static_cast<float>((i % 5)) - 2.0f;
    }
    const auto expected = reference_matmul(shape, K, input_values, weights_data);

    // First inference: distinct input, remote output bound zero-copy (no aliasing yet).
    // A USM-host input is shared, not engine-tracked, so switching inputs below doesn't free tracked memory.
    auto distinct_input = gpu_context.create_usm_host_tensor(ov::element::f32, shape);
    std::copy(input_values.begin(), input_values.end(), static_cast<float*>(distinct_input.get()));
    request.set_input_tensor(distinct_input);
    request.set_output_tensor(remote_output);
    OV_ASSERT_NO_THROW(request.infer());

    // Now feed the same remote tensor back as input: the previously cached zero-copy binding must
    // be dropped so the kernel doesn't read and overwrite the same buffer.
    request.set_input_tensor(remote_output);

    // The stale-binding race has a narrow window (depends on infer-1's binding surviving), so use a
    // higher iteration count than the from-start alias tests to sample it reliably.
    constexpr int kIterations = 64;
    for (int iter = 0; iter < kIterations; ++iter) {
        std::copy(input_values.begin(), input_values.end(), out_buffer);

        OV_ASSERT_NO_THROW(request.infer());

        for (size_t i = 0; i < expected.size(); ++i) {
            ASSERT_NEAR(out_buffer[i], expected[i], 1e-2f)
                << "remote output corrupted after becoming aliased at element " << i << " on iteration " << iter;
        }
    }
}

TEST(TensorTest, smoke_dynamicOutputCallerOwnedUsmHostPartialInputAliasFallsBack) {
    auto core = ov::Core();
    if (!gpu_supports_usm_host_output_sharing(core)) {
        GTEST_SKIP() << "Caller-owned USM-host output sharing requires an iGPU with USM support";
    }

    constexpr size_t K = 1024;
    const ov::Shape shape{2, K};
    std::vector<float> weights_data;
    auto model = makeDynamicMatMulModel(K, weights_data);
    auto compiled_model = core.compile_model(model,
                                             core.get_default_context(ov::test::utils::DEVICE_GPU),
                                             ov::hint::inference_precision(ov::element::f32));
    auto gpu_context = compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();
    auto request = compiled_model.create_infer_request();

    auto usm_allocation = gpu_context.create_usm_host_tensor(ov::element::f32, ov::Shape{shape[0], shape[1] + 1});
    auto* usm_data = static_cast<float*>(usm_allocation.get());
    auto output_ptr = usm_data + 1;
    ov::Tensor input_tensor(ov::element::f32, shape, usm_data);
    ov::Tensor output_tensor(ov::element::f32, shape, output_ptr);

    std::vector<float> input_values(ov::shape_size(shape));
    for (size_t i = 0; i < input_values.size(); ++i) {
        input_values[i] = static_cast<float>((i % 5)) - 2.0f;
    }
    const auto expected = reference_matmul(shape, K, input_values, weights_data);

    request.set_input_tensor(input_tensor);
    request.set_output_tensor(output_tensor);

    constexpr int kIterations = 16;
    for (int iter = 0; iter < kIterations; ++iter) {
        std::copy(input_values.begin(), input_values.end(), usm_data);

        OV_ASSERT_NO_THROW(request.infer());

        auto actual = request.get_output_tensor();
        ASSERT_EQ(actual.data(), output_ptr);
        ASSERT_EQ(actual.get_size(), expected.size());
        const auto* actual_data = actual.data<const float>();
        for (size_t i = 0; i < actual.get_size(); ++i) {
            ASSERT_NEAR(actual_data[i], expected[i], 1e-2f)
                << "partially aliased input/output corrupted at element " << i << " on iteration " << iter;
        }
    }
}

// The output overlap must use the caller buffer's capacity, not the current (possibly shrunk) logical
// shape: after wait() shrinks the output, an input parked in the allocation tail looks disjoint by
// logical span but is overwritten once the output grows back during execution. The alias guard must
// still fall back to a plugin buffer + copy-out. Bind large -> shrink -> park input in tail -> grow.
TEST(TensorTest, smoke_dynamicOutputCallerOwnedUsmHostShrinkThenGrowTailAliasIsSafe) {
    auto core = ov::Core();
    if (!gpu_supports_usm_host_output_sharing(core)) {
        GTEST_SKIP() << "Caller-owned USM-host output sharing requires an iGPU with USM support";
    }

    constexpr size_t K = 1024;
    constexpr size_t capacity_rows = 4;
    std::vector<float> weights_data;
    auto model = makeDynamicMatMulModel(K, weights_data);
    auto compiled_model = core.compile_model(model,
                                             core.get_default_context(ov::test::utils::DEVICE_GPU),
                                             ov::hint::inference_precision(ov::element::f32));
    auto gpu_context = compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();
    auto request = compiled_model.create_infer_request();

    // One allocation of capacity_rows x K; the output view starts at row 0, the aliasing input at row 1.
    auto usm_allocation = gpu_context.create_usm_host_tensor(ov::element::f32, ov::Shape{capacity_rows, K});
    auto* usm_data = static_cast<float*>(usm_allocation.get());

    // Bind the output at full capacity first so its recorded actual_size is capacity_rows x K.
    ov::Tensor output_tensor(ov::element::f32, ov::Shape{capacity_rows, K}, usm_data);
    request.set_output_tensor(output_tensor);

    // Warm-up at capacity, then shrink the output to a single row via a 1-row inference.
    {
        ov::Tensor warm_input(ov::element::f32, ov::Shape{capacity_rows, K});
        std::fill_n(warm_input.data<float>(), warm_input.get_size(), 1.0f);
        request.set_input_tensor(warm_input);
        OV_ASSERT_NO_THROW(request.infer());

        ov::Tensor shrink_input(ov::element::f32, ov::Shape{1, K});
        std::fill_n(shrink_input.data<float>(), shrink_input.get_size(), 1.0f);
        request.set_input_tensor(shrink_input);
        OV_ASSERT_NO_THROW(request.infer());  // wait() shrinks the output tensor to {1, K}
    }

    // Park a 2-row input in the tail (rows 1..3): disjoint from the shrunk 1-row output span, but the
    // output grows to 2 rows during execution and overwrites row 1 unless the guard uses capacity.
    const ov::Shape grow_shape{2, K};
    auto* input_ptr = usm_data + K;  // row 1
    ov::Tensor tail_input(ov::element::f32, grow_shape, input_ptr);
    request.set_input_tensor(tail_input);

    std::vector<float> input_values(ov::shape_size(grow_shape));
    for (size_t i = 0; i < input_values.size(); ++i) {
        input_values[i] = static_cast<float>((i % 5)) - 2.0f;
    }
    const auto expected = reference_matmul(grow_shape, K, input_values, weights_data);

    constexpr int kIterations = 16;
    for (int iter = 0; iter < kIterations; ++iter) {
        std::copy(input_values.begin(), input_values.end(), input_ptr);

        OV_ASSERT_NO_THROW(request.infer());

        auto actual = request.get_output_tensor();
        ASSERT_EQ(actual.data(), usm_data);
        ASSERT_EQ(actual.get_size(), expected.size());
        const auto* actual_data = actual.data<const float>();
        for (size_t i = 0; i < actual.get_size(); ++i) {
            ASSERT_NEAR(actual_data[i], expected[i], 1e-2f)
                << "tail-aliased input corrupted after shrink-then-grow at element " << i << " on iteration " << iter;
        }
    }
}

// AUTO_BATCH's shared buffer must be sized batch=N, not the slot's own batch=1 port. Checked
// by value: an offset bug doesn't change the exposed shape, only which bytes get read/written.
TEST(TensorTest, smoke_lazyAllocAutoBatchUsesBatchedShapeNotSlotShape) {    constexpr int kBatch = 4;
    ov::Shape shape;
    auto model = makeStaticInputModel(shape);

    auto core = ov::Core();
    // Non-zero timeout so requests are actually batched together (same value used in
    // auto_batch/tests/functional/behavior/ov_plugin/auto_batching_tests.cpp).
    auto compiled_model = core.compile_model(model,
                                             "BATCH:" + std::string(ov::test::utils::DEVICE_GPU) +
                                                 "(" + std::to_string(kBatch) + ")",
                                             ov::auto_batch_timeout(1000));

    std::vector<ov::InferRequest> requests;
    for (int i = 0; i < kBatch; ++i) {
        requests.push_back(compiled_model.create_infer_request());
    }

    // Distinct value per slot: an offset bug would mix these together.
    for (int i = 0; i < kBatch; ++i) {
        ov::Tensor in;
        OV_ASSERT_NO_THROW(in = requests[i].get_input_tensor(0));
        ASSERT_EQ(in.get_shape(), shape);
        auto* in_data = in.data<float>();
        ASSERT_NE(in_data, nullptr);
        std::fill_n(in_data, in.get_size(), static_cast<float>(i + 1));
    }

    for (auto& req : requests) {
        OV_ASSERT_NO_THROW(req.start_async());
    }
    for (auto& req : requests) {
        OV_ASSERT_NO_THROW(req.wait());
    }

    // Relu(i+1) == i+1; a corrupted shared buffer would mix values across slots.
    for (int i = 0; i < kBatch; ++i) {
        ov::Tensor out;
        OV_ASSERT_NO_THROW(out = requests[i].get_output_tensor(0));
        ASSERT_EQ(out.get_shape(), shape);
        const auto* data = out.data<float>();
        ASSERT_NE(data, nullptr);
        for (size_t j = 0; j < out.get_size(); ++j) {
            ASSERT_FLOAT_EQ(data[j], static_cast<float>(i + 1)) << "slot " << i << " element " << j;
        }
    }
}

static std::shared_ptr<ov::Model> makeDynamicInputModel() {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                         ov::PartialShape{1, ov::Dimension::dynamic()});
    param->set_friendly_name("dyn_input");
    param->output(0).get_tensor().set_names({"dyn_input"});
    auto relu = std::make_shared<ov::op::v0::Relu>(param);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(relu)};
    return std::make_shared<ov::Model>(results, ov::ParameterVector{param});
}

TEST(TensorTest, smoke_eagerAllocDynamicInputInfer) {
    auto model = makeDynamicInputModel();

    auto core = ov::Core();
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU);
    auto request = compiled_model.create_infer_request();

    // Eagerly allocated: accessing the tensor before set/infer must not throw.
    OV_ASSERT_NO_THROW(request.get_input_tensor(0));

    const ov::Shape concrete{1, 8};
    ov::Tensor input(ov::element::f32, concrete);
    OV_ASSERT_NO_THROW(request.set_input_tensor(0, input));
    OV_ASSERT_NO_THROW(request.infer());

    ov::Tensor out;
    OV_ASSERT_NO_THROW(out = request.get_output_tensor(0));
    ASSERT_EQ(out.get_shape(), concrete);
}

// Many threads race on the first get_tensor() of a fresh request; ensure_input_allocated()'s
// lock must collapse them into a single allocation so all threads observe the same buffer.
TEST(TensorTest, smoke_lazyAllocStaticInputConcurrentFirstGetTensorRace) {
    ov::Shape shape;
    auto model = makeStaticInputModel(shape);

    auto core = ov::Core();
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU);

    constexpr int kThreads = 8;
    constexpr int kIterations = 100;

    std::atomic<bool> failed{false};

    for (int iter = 0; iter < kIterations && !failed.load(); ++iter) {
        // Fresh request re-arms the race for each iteration.
        auto request = compiled_model.create_infer_request();

        std::atomic<int> ready{0};
        std::atomic<bool> go{false};
        std::vector<const void*> observed(kThreads, nullptr);

        std::vector<std::thread> threads;
        threads.reserve(kThreads);
        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&, t] {
                ready.fetch_add(1, std::memory_order_acq_rel);
                while (!go.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                try {
                    auto in = request.get_input_tensor(0);
                    if (in.get_shape() != shape || in.data() == nullptr) {
                        failed.store(true);
                    }
                    observed[t] = in.data();
                } catch (...) {
                    // Any exception on the concurrent first-touch materialization is a failure.
                    failed.store(true);
                }
            });
        }

        // Release all threads at once to maximize contention.
        while (ready.load(std::memory_order_acquire) < kThreads) {
            std::this_thread::yield();
        }
        go.store(true, std::memory_order_release);

        for (auto& th : threads) {
            th.join();
        }

        // All racers must observe the same buffer.
        const void* first = observed[0];
        for (int t = 1; t < kThreads; ++t) {
            if (observed[t] != first) {
                failed.store(true);
            }
        }
    }

    ASSERT_FALSE(failed.load());
}

// Many threads race on the first get_tensor() of a fresh request; lazy output
// materialization must collapse them into a single allocation.
TEST(TensorTest, smoke_lazyAllocStaticOutputConcurrentFirstGetTensorRace) {
    ov::Shape shape;
    auto model = makeStaticInputModel(shape);

    auto core = ov::Core();
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU);

    constexpr int kThreads = 8;
    constexpr int kIterations = 100;

    std::atomic<bool> failed{false};

    for (int iter = 0; iter < kIterations && !failed.load(); ++iter) {
        auto request = compiled_model.create_infer_request();

        std::atomic<int> ready{0};
        std::atomic<bool> go{false};
        std::vector<const void*> observed(kThreads, nullptr);

        std::vector<std::thread> threads;
        threads.reserve(kThreads);
        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&, t] {
                ready.fetch_add(1, std::memory_order_acq_rel);
                while (!go.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                try {
                    auto out = request.get_output_tensor(0);
                    if (out.get_shape() != shape || out.data() == nullptr) {
                        failed.store(true);
                    }
                    observed[t] = out.data();
                } catch (...) {
                    failed.store(true);
                }
            });
        }

        while (ready.load(std::memory_order_acquire) < kThreads) {
            std::this_thread::yield();
        }
        go.store(true, std::memory_order_release);

        for (auto& th : threads) {
            th.join();
        }

        const void* first = observed[0];
        for (int t = 1; t < kThreads; ++t) {
            if (observed[t] != first) {
                failed.store(true);
            }
        }
    }

    ASSERT_FALSE(failed.load());
}

TEST(TensorTest, smoke_canSetShapeForPreallocatedTensor) {
    auto core = ov::Core();
    using namespace ov::preprocess;
    auto p = PrePostProcessor(ov::test::utils::make_split_multi_conv_concat());
    p.input().tensor().set_element_type(ov::element::i8);
    p.input().preprocess().convert_element_type(ov::element::f32);

    auto function = p.build();
    auto exec_net = core.compile_model(function, ov::test::utils::DEVICE_GPU);
    auto inf_req = exec_net.create_infer_request();

    // Check set_shape call for pre-allocated input/output tensors
    auto input_tensor = inf_req.get_input_tensor(0);
    OV_ASSERT_NO_THROW(input_tensor.set_shape({1, 4, 20, 20}));
    OV_ASSERT_NO_THROW(input_tensor.set_shape({1, 3, 20, 20}));
    OV_ASSERT_NO_THROW(input_tensor.set_shape({2, 3, 20, 20}));
    auto output_tensor = inf_req.get_output_tensor(0);
    OV_ASSERT_NO_THROW(output_tensor.set_shape({1, 10, 12, 12}));
    OV_ASSERT_NO_THROW(output_tensor.set_shape({1, 10, 10, 10}));
    OV_ASSERT_NO_THROW(output_tensor.set_shape({2, 10, 20, 20}));
}

TEST(TensorTest, smoke_canSetScalarTensor) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f64, ov::Shape{})};
    params.front()->set_friendly_name("Scalar_1");
    params.front()->output(0).get_tensor().set_names({"scalar1"});

    std::vector<size_t> const_shape = {1};
    auto const1 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, const_shape);
    const1->set_friendly_name("Const_1");
    const1->output(0).get_tensor().set_names({"const1"});
    const1->fill_data(ov::element::i64, 0);

    auto unsqueeze1 = std::make_shared<ov::op::v0::Unsqueeze>(params.front(), const1);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(unsqueeze1)};
    auto model = std::make_shared<ov::Model>(results, params);

    auto core = ov::Core();
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU);
    auto request = compiled_model.create_infer_request();
    double real_data = 1.0;
    ov::Tensor input_data(ov::element::f64, {}, &real_data);
    request.set_tensor("scalar1", input_data);
    OV_ASSERT_NO_THROW(request.infer());
}

TEST(TensorTest, smoke_canSetTensorForDynamicInput) {
    auto core = ov::Core();
    using namespace ov::preprocess;
    auto p = PrePostProcessor(ov::test::utils::make_split_multi_conv_concat());
    p.input().tensor().set_element_type(ov::element::i8);
    p.input().preprocess().convert_element_type(ov::element::f32);

    auto function = p.build();
    std::map<size_t, ov::PartialShape> shapes = { {0, ov::PartialShape{-1, -1, -1, -1}} };
    function->reshape(shapes);
    auto exec_net = core.compile_model(function, ov::test::utils::DEVICE_GPU);
    auto inf_req = exec_net.create_infer_request();

    ov::Tensor t1(ov::element::i8, {1, 4, 20, 20});
    ov::Tensor t2(ov::element::i8, {1, 4, 30, 30});
    ov::Tensor t3(ov::element::i8, {1, 4, 40, 40});

    // Check set_shape call for pre-allocated input/output tensors
    OV_ASSERT_NO_THROW(inf_req.set_input_tensor(t1));
    OV_ASSERT_NO_THROW(inf_req.infer());

    OV_ASSERT_NO_THROW(inf_req.set_input_tensor(t2));
    OV_ASSERT_NO_THROW(inf_req.infer());

    OV_ASSERT_NO_THROW(inf_req.set_input_tensor(t3));
    OV_ASSERT_NO_THROW(inf_req.infer());

    OV_ASSERT_NO_THROW(inf_req.set_input_tensor(t3));
    OV_ASSERT_NO_THROW(inf_req.infer());

    OV_ASSERT_NO_THROW(inf_req.set_input_tensor(t1));
    OV_ASSERT_NO_THROW(inf_req.infer());

    OV_ASSERT_NO_THROW(inf_req.set_input_tensor(t2));
    OV_ASSERT_NO_THROW(inf_req.infer());
}

TEST(TensorTest, smoke_canSetTensorForDynamicOutput) {
    auto core = ov::Core();
    using namespace ov::preprocess;
    auto p = PrePostProcessor(ov::test::utils::make_split_multi_conv_concat());
    p.input().tensor().set_element_type(ov::element::i8);
    p.input().preprocess().convert_element_type(ov::element::f32);

    auto function = p.build();
    std::map<size_t, ov::PartialShape> shapes = { {0, ov::PartialShape{-1, -1, -1, -1}} };
    function->reshape(shapes);
    auto exec_net = core.compile_model(function, ov::test::utils::DEVICE_GPU);
    auto inf_req = exec_net.create_infer_request();

    ov::Tensor t1(ov::element::i8, {1, 4, 20, 20});
    auto out_tensor = inf_req.get_output_tensor();
    ov::Tensor t2(out_tensor.get_element_type(), out_tensor.get_shape());
    ASSERT_EQ(t2.get_byte_size(), 0);
    // Check set_shape call for pre-allocated input/output tensors
    OV_ASSERT_NO_THROW(inf_req.set_input_tensor(t1));
    OV_ASSERT_NO_THROW(inf_req.set_output_tensor(t2));
    OV_ASSERT_NO_THROW(inf_req.infer());
    ASSERT_NE(t2.get_byte_size(), 0);
}

TEST(TensorTest, smoke_canReallocateDeviceInputForHostTensor) {
    auto ov = ov::Core();
    using namespace ov::preprocess;
    auto p = PrePostProcessor(ov::test::utils::make_split_multi_conv_concat());
    p.input().tensor().set_element_type(ov::element::i8);
    p.input().preprocess().convert_element_type(ov::element::f32);
    auto function = p.build();

    auto compiled_model = ov.compile_model(function, ov::test::utils::DEVICE_GPU);
    auto inf_req = compiled_model.create_infer_request();

    auto input = function->input();
    ov::Tensor host_tensor(input.get_element_type(), input.get_shape());

    // Infer with pre-allocated input tensor
    OV_ASSERT_NO_THROW(inf_req.infer());

    // Infer with host_tensor
    OV_ASSERT_NO_THROW(inf_req.set_input_tensor(host_tensor));
    OV_ASSERT_NO_THROW(inf_req.infer());
}

TEST(VariablesTest, smoke_canSetStateTensor) {
    auto ov = ov::Core();
    const ov::Shape virable_shape = {1, 3, 2, 4};
    const ov::Shape input_shape = {1, 3, 2, 4};
    const ov::element::Type et = ov::element::f16;
    auto model = ov::test::utils::make_read_concat_split_assign(input_shape, et);
    auto compiled_model = ov.compile_model(model, ov::test::utils::DEVICE_GPU);
    auto request = compiled_model.create_infer_request();

    ov::Tensor variable_tensor(et, virable_shape);
    ov::Tensor input_tensor(et, input_shape);

    auto variables = request.query_state();
    ASSERT_EQ(variables.size(), 1);
    auto variable = variables.front();
    ASSERT_EQ(variable.get_name(), "v0");
    auto default_state_tensor = variable.get_state();
    ASSERT_EQ(default_state_tensor.get_shape(), virable_shape);

    OV_ASSERT_NO_THROW(request.infer());
}

TEST(VariablesTest, smoke_set_get_state_with_convert) {
    auto build_model = [](ov::element::Type type, const ov::PartialShape& shape) {
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        const ov::op::util::VariableInfo variable_info { shape, type, "v0" };
        auto variable = std::make_shared<ov::op::util::Variable>(variable_info);
        auto read_value = std::make_shared<ov::op::v6::ReadValue>(param, variable);
        auto add = std::make_shared<ov::op::v1::Add>(read_value, param);
        auto assign = std::make_shared<ov::op::v6::Assign>(add, variable);
        auto res = std::make_shared<ov::op::v0::Result>(add);
        return std::make_shared<ov::Model>(ov::ResultVector { res }, ov::SinkVector { assign }, ov::ParameterVector{param}, "StateTestModel");
    };

    auto ov = ov::Core();
    const ov::Shape virable_shape = {1, 3, 2, 4};
    const ov::Shape input_shape = {1, 3, 2, 4};
    const ov::element::Type et = ov::element::f32;
    auto model = build_model(et, input_shape);
    auto compiled_model = ov.compile_model(model, ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(ov::element::f16));
    auto request = compiled_model.create_infer_request();

    auto variables = request.query_state();
    ASSERT_EQ(variables.size(), 1);
    auto variable = variables.front();
    ASSERT_EQ(variable.get_name(), "v0");
    auto state_tensor = variable.get_state();
    ASSERT_EQ(state_tensor.get_shape(), virable_shape);
    ASSERT_EQ(state_tensor.get_element_type(), et);

    auto tensor_to_set = ov::test::utils::create_and_fill_tensor(et, state_tensor.get_shape());
    variable.set_state(tensor_to_set);
    state_tensor = variable.get_state();

    ov::test::utils::compare(tensor_to_set, state_tensor, 1e-5f, 1e-5f);
}

TEST(VariablesTest, smoke_padded_tensor_set_get_state_with_convert) {
    auto build_model = [](ov::element::Type type, const ov::PartialShape& shape) {
        auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
        const ov::op::util::VariableInfo variable_info { shape, type, "v0" };
        auto variable = std::make_shared<ov::op::util::Variable>(variable_info);
        auto read_value = std::make_shared<ov::op::v6::ReadValue>(param, variable);
        auto add = std::make_shared<ov::op::v1::Add>(read_value, param);
        auto assign = std::make_shared<ov::op::v6::Assign>(add, variable);
        auto res = std::make_shared<ov::op::v0::Result>(add);
        return std::make_shared<ov::Model>(ov::ResultVector { res }, ov::SinkVector { assign }, ov::ParameterVector{param}, "StateTestModel");
    };

    auto ov = ov::Core();
    const ov::Shape virable_shape_padded = {1, 3, 4, 4};
    const ov::Shape virable_shape = {1, 3, 2, 4};
    const ov::Shape input_shape = {1, 3, 2, 4};
    const ov::element::Type et = ov::element::f32;
    auto model = build_model(et, input_shape);
    auto compiled_model = ov.compile_model(model, ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(ov::element::f16));
    auto request = compiled_model.create_infer_request();

    auto variables = request.query_state();
    ASSERT_EQ(variables.size(), 1);
    auto variable = variables.front();
    ASSERT_EQ(variable.get_name(), "v0");
    auto state_tensor = variable.get_state();
    ASSERT_EQ(state_tensor.get_shape(), virable_shape);
    ASSERT_EQ(state_tensor.get_element_type(), et);

    auto tensor_to_set_padded = ov::test::utils::create_and_fill_tensor(et, virable_shape_padded);

    // trim original tensor
    auto tensor_to_set =
        ov::Tensor(tensor_to_set_padded, ov::Coordinate{0, 0, 0, 0}, ov::Coordinate(virable_shape));

    variable.set_state(tensor_to_set);
    state_tensor = variable.get_state();

    auto res_tensor_ptr = static_cast<float*>(state_tensor.data());
    auto ref_tensor_ptr = static_cast<float*>(tensor_to_set.data());
    auto ref_stride = tensor_to_set.get_strides();
    auto res_stride = state_tensor.get_strides();
    for (size_t i = 0; i < ref_stride.size(); ++i) {
        ref_stride[i] /= (tensor_to_set.get_element_type().bitwidth()/8);
        res_stride[i] /= (state_tensor.get_element_type().bitwidth()/8);
    }
    // ref stride: [48, 16, 4, 1]
    // res stride: [24, 8, 4, 1]
    // compare actual tensor w/o pad
    for (size_t b = 0; b < virable_shape[0]; ++b) {
        for (size_t f = 0; f < virable_shape[1]; ++f) {
            for (size_t y = 0; y < virable_shape[2]; ++y) {
                for (size_t x = 0; x < virable_shape[3]; ++x) {
                    auto ref_idx = b * ref_stride[0] + f * ref_stride[1] + y * ref_stride[2] + x * ref_stride[3];
                    auto res_idx = b * res_stride[0] + f * res_stride[1] + y * res_stride[2] + x * res_stride[3];
                    ASSERT_EQ(res_tensor_ptr[res_idx], ref_tensor_ptr[ref_idx]);
                }
            }
        }
    }
}

#if defined(_WIN32)
// Issue: 126388
TEST(TensorTest, DISABLED_outputTensorShapesForDynamicInput) {
#else
TEST(TensorTest, smoke_outputTensorShapesForDynamicInput) {
#endif
    auto core = ov::Core();
    using namespace ov::preprocess;
    auto p = PrePostProcessor(ov::test::utils::make_split_multi_conv_concat());
    p.input().tensor().set_element_type(ov::element::i8);
    p.input().preprocess().convert_element_type(ov::element::f32);

    auto function = p.build();
    std::map<size_t, ov::PartialShape> shapes = { {0, ov::PartialShape{-1, -1, -1, -1}} };
    function->reshape(shapes);
    auto exec_net = core.compile_model(function, ov::test::utils::DEVICE_GPU);
    auto inf_req = exec_net.create_infer_request();

    ov::Tensor t1(ov::element::i8, {1, 4, 20, 40});
    ov::Tensor t2(ov::element::i8, {1, 4, 40, 20});
    ov::Tensor t3(ov::element::i8, {1, 4, 20, 40});
    const ov::Shape output1_shape = {1, 10, 12, 32};
    const ov::Shape output2_shape = {1, 10, 32, 12};
    const ov::Shape output3_shape = {1, 10, 12, 32};

    // Check output shape of output tensor is correct
    OV_ASSERT_NO_THROW(inf_req.set_input_tensor(t1));
    OV_ASSERT_NO_THROW(inf_req.infer());
    ASSERT_EQ(inf_req.get_output_tensor().get_shape(), output1_shape);

    OV_ASSERT_NO_THROW(inf_req.set_input_tensor(t2));
    OV_ASSERT_NO_THROW(inf_req.infer());
    ASSERT_EQ(inf_req.get_output_tensor().get_shape(), output2_shape);

    OV_ASSERT_NO_THROW(inf_req.set_input_tensor(t3));
    OV_ASSERT_NO_THROW(inf_req.infer());
    ASSERT_EQ(inf_req.get_output_tensor().get_shape(), output3_shape);
}

TEST(TensorTest, smoke_canShareTensorIfModelsFromDifferentCores) {
    auto core1 = ov::Core();
    auto core2 = ov::Core();

    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{4, 8});
    auto relu = std::make_shared<ov::op::v0::Relu>(param);
    auto result = std::make_shared<ov::op::v0::Result>(relu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});

    auto compiled_model1 = core1.compile_model(model, ov::test::utils::DEVICE_GPU);
    auto compiled_model2 = core2.compile_model(model, ov::test::utils::DEVICE_GPU);

    auto request1 = compiled_model1.create_infer_request();
    auto request2 = compiled_model2.create_infer_request();

    request2.set_input_tensor(request1.get_output_tensor());
    request2.set_output_tensor(request1.get_input_tensor());

    OV_ASSERT_NO_THROW(request1.infer());
    OV_ASSERT_NO_THROW(request2.infer());
}

TEST(CoreTest, smoke_singletonOclContext) {
    auto core1 = ov::Core();
    auto ctx1 = core1.get_default_context("GPU");
    auto& oclContext1 = static_cast<ov::intel_gpu::ocl::ClContext&>(ctx1);
    auto core2 = ov::Core();
    auto ctx2 = core2.get_default_context("GPU");
    auto& oclContext2 = static_cast<ov::intel_gpu::ocl::ClContext&>(ctx2);
    ASSERT_EQ(oclContext1.get(), oclContext2.get());
}

} // namespace
