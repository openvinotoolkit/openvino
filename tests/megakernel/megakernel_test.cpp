#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/runtime/properties.hpp"
#include "TestMegaKernel/megakernelImpl.h"

namespace {
void set_megakernel_disabled(const char* value) {
#ifdef _WIN32
    _putenv_s("OV_MEGAKERNEL_DISABLE", value ? value : "");
#else
    if (value)
        setenv("OV_MEGAKERNEL_DISABLE", value, 1);
    else
        unsetenv("OV_MEGAKERNEL_DISABLE");
#endif
}

std::shared_ptr<ov::Model> make_model(bool matching = true) {
    auto left = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 1, 32});
    auto right = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 1, 32});
    auto add = std::make_shared<ov::op::v1::Add>(left, right);
    add->set_friendly_name(matching ? "test_megakernel_add" : "other_add");
    auto relu = std::make_shared<ov::op::v0::Relu>(add);
    relu->set_friendly_name("test_megakernel_relu");
    return std::make_shared<ov::Model>(ov::OutputVector{relu}, ov::ParameterVector{left, right});
}

class DisableMegakernel {
public:
    explicit DisableMegakernel(bool disable) {
        const char* current = std::getenv("OV_MEGAKERNEL_DISABLE");
        if (current) {
            existed = true;
            value = current;
        }
        set_megakernel_disabled(disable ? "1" : "0");
    }
    ~DisableMegakernel() {
        set_megakernel_disabled(existed ? value.c_str() : nullptr);
    }
private:
    bool existed = false;
    std::string value;
};

ov::CompiledModel compile(ov::Core& core, bool disable, bool matching = true) {
    DisableMegakernel scoped_disable(disable);
    return core.compile_model(make_model(matching), "GPU", ov::hint::inference_precision(ov::element::f32));
}

std::vector<float> infer(ov::CompiledModel& model, int iteration) {
    auto request = model.create_infer_request();
    auto left = request.get_input_tensor(0);
    auto right = request.get_input_tensor(1);
    for (size_t index = 0; index < left.get_size(); ++index) {
        left.data<float>()[index] = static_cast<float>(static_cast<int>(index) - 16 + iteration);
        right.data<float>()[index] = static_cast<float>(static_cast<int>(index % 5) - 2);
    }
    request.infer();
    auto output = request.get_output_tensor();
    return {output.data<float>(), output.data<float>() + output.get_size()};
}
}

TEST(MegaKernelIntegration, MatchesUnfusedGpuAndRunsBackend) {
    ov::Core core;
    auto baseline = compile(core, true);
    auto fused = compile(core, false);
    const auto before = TestMegaKernelExecutionCount();
    for (int iteration = 0; iteration < 3; ++iteration) {
        auto expected = infer(baseline, iteration);
        EXPECT_EQ(TestMegaKernelExecutionCount(), before + static_cast<unsigned int>(iteration));
        auto actual = infer(fused, iteration);
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t index = 0; index < actual.size(); ++index) {
            EXPECT_FLOAT_EQ(actual[index], expected[index]) << "index " << index;
            const auto sum = static_cast<int>(index) - 16 + iteration + static_cast<int>(index % 5) - 2;
            EXPECT_FLOAT_EQ(actual[index], std::max(0.0f, static_cast<float>(sum)));
        }
        EXPECT_EQ(TestMegaKernelExecutionCount(), before + static_cast<unsigned int>(iteration) + 1);
    }
}

TEST(MegaKernelIntegration, DoesNotFuseUnrelatedGraph) {
    ov::Core core;
    auto model = compile(core, false, false);
    const auto before = TestMegaKernelExecutionCount();
    const auto output = infer(model, 0);
    EXPECT_EQ(output.size(), 32);
    EXPECT_EQ(TestMegaKernelExecutionCount(), before);
}