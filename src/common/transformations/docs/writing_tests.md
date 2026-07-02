# Writing Tests for OpenVINO Transformations

## Contents

- [Test location and build target](#test-location-and-build-target)
- [Quick checklist](#quick-checklist)
- [Test class setup](#test-class-setup)
- [FunctionsComparator flags](#functionscomparator-flags)
- [Node builders](#node-builders)
- [Model builder helpers](#model-builder-helpers)
- [Parametrized tests](#parametrized-tests)
- [Negative tests (transformation must NOT fire)](#negative-tests-transformation-must-not-fire)
- [What NOT to do](#what-not-to-do)

## Test location and build target

Test files live in `src/common/transformations/tests/` with subdirectories by category (e.g., `common_optimizations/`, `utils/`). Build and run all transformation tests with:

```bash
cmake --build build --target ov_transformations_tests -j$(nproc)
./bin/ov_transformations_tests
```

Or run a single test:
```bash
./bin/ov_transformations_tests --gtest_filter="YourTestName"
```

## Quick checklist

1. Base class: `TransformationTestsF` (not `TransformationTests`)
2. Enable only the `FunctionsComparator::CmpValues` flags relevant to the transformation
3. Build models in helper functions to eliminate duplication
4. Parametrize when multiple input shapes / configs exercise the same structural pattern
5. Consider using `node_builders/` helpers instead of directly instantiating ops when a builder exists and improves test readability
6. Avoid building `model_ref` when the transformation does not modify the `model`, since `model_ref` is initialized as a clone of `model` by default

## Test class setup

```cpp
#include "common_test_utils/ov_test_utils.hpp"   // TransformationTestsF, FunctionsComparator

// Minimal fixture — use TEST_F when no parameters are needed
class MyTransformTests : public TransformationTestsF {
public:
    MyTransformTests() {
        // Enable additional comparator flags if your transformation changes them (e.g. CONST_VALUES, ATTRIBUTES).
        // Skip this constructor if the default flags suffice.
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ov::pass::MyTransformation>();
    }
};

TEST_F(MyTransformTests, MyTransformName_SomeVariant) {
    {   // original model
        auto data = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{1, 3, 16, 16});
        // ... build graph ...
        model = std::make_shared<ov::Model>(output, ov::ParameterVector{data});
    }
    {   // reference model — what the transformation should produce
        auto data = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::Shape{1, 3, 16, 16});
        // ...
        model_ref = std::make_shared<ov::Model>(output_ref, ov::ParameterVector{data});
    }
}
```

`TransformationTestsF` automatically:
- Runs `InitNodeInfo` before the registered pass
- Runs `FunctionsComparator` comparison in `TearDown`

**`manager.register_pass<*>()` should be moved to fixture's `SetUp()`** when multiple TEST_F tests in the same fixture use the same transformation — this avoids duplication and makes the transformation explicit per fixture rather than hidden in each test body.

**Never** manually create a `pass::Manager`, call `check_rt_info`, or call `compare_functions` — those are handled by the fixture.

## FunctionsComparator flags

`TransformationTestsF` initialises the comparator with `no_default()` and then explicitly enables `NODES | PRECISIONS | RUNTIME_KEYS | SUBGRAPH_DESCRIPTORS`.

Enable **additional** flags only when they matter for correctness of your test:

| Flag | On by default in `TransformationTestsF`? | When to add |
|------|------------------------------------------|-------------|
| `NODES` | yes | (always on) op type and graph topology |
| `PRECISIONS` | yes | (always on) element type of each tensor |
| `RUNTIME_KEYS` | yes | (always on) `rt_info` key presence and values |
| `SUBGRAPH_DESCRIPTORS` | yes | (always on) SubGraphOp bodies port descriptors |
| `CONST_VALUES` | no | Transformation changes, inserts, or folds constant data |
| `ATTRIBUTES` | no | Transformation sets or changes op attributes (broadcast mode, strides, group count, …), or creates new nodes with some specific attributes |
| `NAMES` | no | Transformation must preserve specific friendly names |
| `TENSOR_NAMES` | no | Transformation must preserve output tensor names |
| `CONSUMERS_COUNT` | no | Transformation changes the number of consumers on an output (e.g. constant deduplication, shared subgraph nodes) or the transformation's logic depends on the consumers count |
| `ACCURACY` | no | Runs Template plugin inference on the pre- and post-transformation models and checks outputs are within threshold (structural `model_ref` comparison still runs after). **Makes tests significantly slower** — only enable when there is a direct justification; plugin tests already verify accuracy for most transformations |

Enable in the constructor or `SetUp`, or inline at the test level:

```cpp
// In a parametrized class constructor:
comparator.enable(FunctionsComparator::ATTRIBUTES);

// In an individual TEST_F (after both models are assigned):
comparator.enable(FunctionsComparator::CONST_VALUES);
```

Do **not** explicitly enable flags already on by default in `TransformationTestsF` (`NODES`, `PRECISIONS`, `RUNTIME_KEYS`, `SUBGRAPH_DESCRIPTORS`).

## Node builders

Use `node_builders/` helpers instead of constructing ops directly when a builder exists and improves test readability.
Headers live in `src/tests/test_utils/common_test_utils/include/common_test_utils/node_builders/`.

Import example:
```cpp
#include "common_test_utils/node_builders/eltwise.hpp"
// usage:
auto add = ov::test::utils::make_eltwise(data, constant, ov::test::utils::EltwiseTypes::ADD);
```

## Model builder helpers

Extract original and reference model construction into functions (or static methods) to avoid duplicating parameter/shape handling. Name them `getModel` and `getModelRef`.

Use a **single combined builder** only when the two models are nearly identical — e.g., they differ only in a constant value or a single attribute — and the shared structure would be more confusing to duplicate than to branch:

```cpp
// Single builder only when divergence is minimal (e.g. one constant differs):
std::shared_ptr<ov::Model> getModel(const ov::PartialShape& shape, float alpha, bool addBias) {
    auto data = std::make_shared<ov::opset10::Parameter>(ov::element::f32, shape);
    auto c    = ov::opset10::Constant::create(ov::element::f32, {1}, {alpha});
    auto mul  = std::make_shared<ov::opset10::Multiply>(data, c);
    if (addBias) {
        auto bias = ov::opset10::Constant::create(ov::element::f32, {1}, {0.f});
        return std::make_shared<ov::Model>(
            std::make_shared<ov::opset10::Add>(mul, bias), ov::ParameterVector{data});
    }
    return std::make_shared<ov::Model>(mul, ov::ParameterVector{data});
}
model     = getModel(shape, 0.5f, /*addBias=*/true);
model_ref = getModel(shape, 0.5f, /*addBias=*/false);
```

When the two models have **different op topologies**, use separate `getModel` / `getModelRef` functions — do not force them into one builder with a `bool reference` flag:

```cpp
std::shared_ptr<ov::Model> getModel(const ov::PartialShape& shape) {
    auto data = std::make_shared<ov::opset10::Parameter>(ov::element::f32, shape);
    auto a    = std::make_shared<ov::opset10::OpA>(data);
    auto b    = std::make_shared<ov::opset10::OpB>(a);
    return std::make_shared<ov::Model>(b, ov::ParameterVector{data});
}

std::shared_ptr<ov::Model> getModelRef(const ov::PartialShape& shape) {
    auto data = std::make_shared<ov::opset10::Parameter>(ov::element::f32, shape);
    auto op   = std::make_shared<ov::opset10::SomeOp>(data);
    return std::make_shared<ov::Model>(op, ov::ParameterVector{data});
}
```

## Parametrized tests

Use `testing::WithParamInterface<ParamType>` + `INSTANTIATE_TEST_SUITE_P` when multiple shapes, precisions, op variants, or broadcast modes all exercise the same structural transform.

```cpp
using MyTestParams = std::tuple<ov::element::Type, ov::Shape, /*…*/>;

class MyTransformTests : public testing::WithParamInterface<MyTestParams>,
                         public TransformationTestsF {
public:
    MyTransformTests() : TransformationTestsF() {
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
    static std::string getTestCaseName(const testing::TestParamInfo<MyTestParams>& info) {
        const auto& [prc, shape] = info.param;
        std::ostringstream ss;
        ss << "prc=" << prc << "_shape=" << shape;
        return ss.str();
    }
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        const auto& [prc, shape] = GetParam();
        model     = getModel(prc, shape);
        model_ref = getModelRef(prc, shape);
        manager.register_pass<ov::pass::MyTransformation>();
    }
};

TEST_P(MyTransformTests, Transform) {}

INSTANTIATE_TEST_SUITE_P(MyTransformTests, MyTransformTests,
    ::testing::Combine(
        ::testing::Values(ov::element::f32, ov::element::f16),
        ::testing::Values(ov::Shape{1,3,16,16}, ov::Shape{1,1,4,4})),
    MyTransformTests::getTestCaseName);
```

Corner cases whose model structure significantly differs from the parametrized builder belong in standalone `TEST_F` tests.

## Negative tests (transformation must NOT fire)

When the transformation should leave the `model` unchanged, leave `model_ref` as `nullptr` — the TransformationTestsF class logic assigns a copy of `model` to `model_ref`. The cleanest pattern:

```cpp
TEST_F(TransformationTestsF, MyTransform_Negative_SomeCondition) {
    model = buildOriginal(…);
    manager.register_pass<ov::pass::MyTransformation>();
    // model_ref intentionally omitted
}
```

## What NOT to do

- Do **not** inherit from plain `TransformationTests` / `ov::test::TestsCommon`
- Do **not** manually instantiate `pass::Manager` or call `manager.run_passes(model)`
- Do **not** call `check_rt_info(f)` or the deprecated `compare_functions(f, f_ref, …)`
- Do **not** add flags already on by default in `TransformationTestsF` (`NODES`, `PRECISIONS`, `RUNTIME_KEYS`, `SUBGRAPH_DESCRIPTORS`)
