# Covering Low Precision Transformations (LPT) with Transformation Tests

## Introduction

This instruction provides a comprehensive guide on how to add new tests to the existing Low Precision Transformations (LPT) test framework in OpenVINO.

The LPT tests are designed to verify that low-precision transformations work correctly by comparing the actual model transformation results against expected outcomes. These tests use specialized data structures and builder patterns that make it easy to construct test scenarios and validate transformation behavior.

## Overview

The LPT test framework employs a structured approach where each transformation test follows a consistent pattern:

1. **Test Structure**: Each test case is defined using a `*TransformationTestValues` structure that usually contains test parameters and both actual (input) and expected (output) model states.

2. **Model Building**: Test models are constructed using specialized builder classes from the `ov::builder::subgraph` namespace that provide convenient methods for creating quantized models.

3. **Transformation Execution**: The test framework applies transformations to the actual model and compares the result with the expected model structure.

## Key Components

This guide will cover the following essential components:

- **TestTransformationParams**: Configuration parameters that control how transformations are applied
- **Actual and Expected Substructures**: Input and output model states for test validation  
- **LPT Builder Structures**: Specialized classes for building quantized test models
- **Model Building Functions**: Helper functions that construct test graphs using LPT builders

Understanding these components will enable you to create comprehensive test cases that properly validate low-precision transformation behavior in various scenarios.

---

## 1. TestTransformationParams Structure

The `TestTransformationParams` structure is the core configuration class that controls how LPT are applied during testing: it is needed to emulate plugin configuration.

> **⚠️ Important Note:** These parameters only affect how LPT passes are executed during transformation. They do **not** influence the test model creation itself - the input model structure is built separately using the Actual substructure and builder functions.

### Key Members

**Definition**: [layer_transformation.hpp](./layer_transformation.hpp)

```cpp
struct TestTransformationParams {
    bool updatePrecisions;                                  // Whether to update precision (from float precision to low precision) during transformation
    std::vector<ov::element::Type> precisionsOnActivations; // Allowed precisions for activations (e.g., u8, i8)
    std::vector<ov::element::Type> precisionsOnWeights;     // Allowed precisions for weights (e.g., i8)
    bool supportAsymmetricQuantization;                     // Support for asymmetric quantization
    ov::element::Type deqPrecision;                         // Dequantization precision (usually f32)
    bool deconvolutionSpecificChannelsRatio;                // Specific for deconvolution operations
    std::vector<ov::element::Type> defaultPrecisions;       // Default supported precisions
};
```

### Common Parameter Configurations

The framework provides several predefined parameter configurations:

**Implementation**: [layer_transformation.cpp](./layer_transformation.cpp)

- **`LayerTransformation::createParamsU8I8()`** - U8 precision on activations, I8 precision on weights, requested by the plugin
- **`LayerTransformation::createParamsI8I8()`** - I8 precision on activations, I8 precision on weights, requested by the plugin
- **`LayerTransformation::createParamsU8U8()`** - U8 precision on activations, U8 precision on weights, requested by the plugin

### Usage Example

```cpp
// Create standard U8/I8 parameters
TestTransformationParams params = LayerTransformation::createParamsU8I8();

// Customize parameters for specific test case
params.setSupportAsymmetricQuantization(false)
      .setUpdatePrecisions(true)
      .setDeqPrecision(ov::element::f32);
```

## 2. Actual and Expected Substructures

Each transformation test defines both the input state (Actual) and the expected output state (Expected) of the model. These substructures describe the model graph before and after transformation.

### Structure Pattern

```cpp
class TransformationTestValues {
public:
    class Actual {
        ov::element::Type precisionBeforeDequantization;  // Input tensor precision
        ov::builder::subgraph::DequantizationOperations dequantization;  // Dequantization operations (Convert→Subtract→Multiply chain) before a target operation
    };

    class Expected {
        ov::element::Type precisionBeforeDequantization;  // Input tensor precision
        ov::builder::subgraph::DequantizationOperations dequantizationBefore;  // Expected dequantization before the target operation
        ov::element::Type precisionAfterOperation;        // Expected precision after the target operation
        ov::builder::subgraph::DequantizationOperations dequantizationAfter;   // Expected dequantization after the target operation
    };

    TestTransformationParams params;  // Transformation configuration parameters
    Actual actual;                   // Input model description
    Expected expected;              // Expected output model description
};
```

### Example Usage

```cpp
// Define test values for Clamp transformation (from ClampTransformationTestValues)
ClampTransformationTestValues testValues = {
    LayerTransformation::createParamsU8I8(),  // params - use standard U8/I8 configuration
    {   // actual - input model state
        ov::element::u8,                      // precisionBeforeDequantization
        {{ov::element::f32}, {128.f}, {3.f}} // dequantization: Convert(u8→f32), Subtract(128), Multiply(3.f)
    },
    {   // expected - expected output model state after transformation
        ov::element::u8,                      // precisionBeforeDequantization 
        {{}, {}, {}},                         // dequantizationBefore (empty - no dequant before operation)
        ov::element::f32,                     // precisionAfterOperation
        {{}, {128.f}, {3.f}}                  // dequantizationAfter: dequant moved after Clamp operation
    }
};
```

> **📝 Note:** The `DequantizationOperations` structure and its configuration options are described in detail in the next section.

## 3. Typical LPT Builder Structures

The LPT test framework provides specialized builder classes to construct quantized models easily. These are located in the `ov::builder::subgraph` namespace.

This section focuses on the two most commonly used builder structures:
- **`DequantizationOperations`** - for describing Convert→Subtract→Multiply chains
- **`FakeQuantizeOnData`** - for describing FakeQuantize operation parameters

To understand where these test structures map to actual model operations, consider the following example model:

![FakeQuantize and Convolution Model](../../../../docs/articles_en/assets/images/model_fq_and_convolution.common.svg)

In this diagram:
- **`FakeQuantizeOnData`** structure corresponds to the **FakeQuantize** operation on activations (input data)
- **`DequantizationOperations`** structure corresponds to the **dequantization operations on weights** (Convert→Subtract→Multiply chain)

The following sections provide detailed information on how these test structures are built and configured.

### DequantizationOperations

The main class for describing dequantization operations.

**Definition**: [dequantization_operations.hpp](../../../tests/ov_helpers/ov_lpt_models/include/ov_lpt_models/common/dequantization_operations.hpp)

```cpp
class DequantizationOperations {
public:
    class Convert {
        ov::element::Type outPrecision;     // Output precision (e.g., f32)
        bool addDequantizationAttribute;    // Add Dequantization attribute
    };
    
    class Subtract {
        std::vector<float> values;          // Subtract values (zero points)
        ov::element::Type outPrecision;     // Output precision  
        ov::Shape constantShape;            // Shape of constant
        size_t constantIndex;               // Input index for constant (usually 1)
        ov::element::Type constantPrecision; // Precision of constant
    };
    
    class Multiply {
        std::vector<float> values;          // Scale factors
        ov::element::Type outPrecision;     // Output precision
        ov::Shape constantShape;            // Shape of constant  
        size_t constantIndex;               // Input index for constant (usually 1)
        ov::element::Type constantPrecision; // Precision of constant
    };

    Convert convert;
    Subtract subtract;  
    Multiply multiply;
};
```

#### Basic Dequantization Patterns

```cpp
// 1. Simple per-tensor dequantization: Convert(f32) → Subtract(128) → Multiply(0.02)  
DequantizationOperations dequant = {
    {ov::element::f32},  // Convert
    {128.f},             // Subtract
    {0.02f}              // Multiply
};

// 2. Only scaling (no zero point): Convert(f32) → Multiply(0.5)
DequantizationOperations scaleOnly = {
    {ov::element::f32},  // Convert
    {},                  // Subtract (empty)
    {0.5f}               // Multiply
};

// 3. Empty dequantization (no operations)
DequantizationOperations empty = {{}, {}, {}};

// 4. Mixed initialization with method chaining for precision configuration
DequantizationOperations mixedPrecision = {
    {},  // Convert (empty)
    {},  // Subtract (empty) 
    // Single value multiply with f16 output precision and f32 constant precision
    DequantizationOperations::Multiply({3.f}, ov::element::f16).setConstantPrecision(ov::element::f32)
};
```

#### Per-Channel Dequantization

```cpp
// Per-channel dequantization with different zero points and scales per channel
DequantizationOperations perChannel = {
    {ov::element::f32},                    // Convert
    {{128.f, 0.f, 64.f}},                // Subtract - different zero points per channel
    {{0.01f, 0.02f, 0.015f}}             // Multiply - different scales per channel
};
```

#### Advanced Constructor Options

```cpp
// Subtract with explicit shape and precision configuration
DequantizationOperations::Subtract subtractAdvanced(
    {128.f, 0.f, 64.f},           // values - zero points
    ov::element::f32,             // outPrecision - output precision after subtract
    {1, 3, 1, 1},                // constantShape - shape of subtract constant
    false,                        // toRemove - don't remove this operation
    1ul,                          // constantIndex - input index for constant (usually 1)
    ov::element::f32,             // constantPrecision - precision of the constant
    false                         // addConvert - don't add convert between subtract constant and subtract
);

// Multiply with custom shape and index
DequantizationOperations::Multiply multiplyAdvanced(
    {0.01f, 0.02f, 0.015f},       // values - scale factors
    ov::element::f32,             // outPrecision
    {1, 3, 1, 1},                // constantShape
    false,                        // toRemove
    1ul,                          // constantIndex
    ov::element::f32,             // constantPrecision
    false                         // addConvert
);

// Combine advanced operations
DequantizationOperations advancedDequant(
    DequantizationOperations::Convert(ov::element::f32),
    subtractAdvanced,
    multiplyAdvanced
);
```

### FakeQuantizeOnData

The main class for describing FakeQuantize operation parameters and configuration.

**Definition**: [fake_quantize_on_data.hpp](../../../tests/ov_helpers/ov_lpt_models/include/ov_lpt_models/common/fake_quantize_on_data.hpp)

```cpp
class FakeQuantizeOnData {
    uint64_t quantizationLevel;           // Quantization levels (usually 256 for 8-bit)
    ov::Shape constantShape;              // Shape of quantization parameters
    std::vector<float> inputLowValues;    // Input low values
    std::vector<float> inputHighValues;   // Input high values  
    std::vector<float> outputLowValues;   // Output low values
    std::vector<float> outputHighValues;  // Output high values
    ov::element::Type outputPrecision;    // Output precision (u8, i8, etc.)
};
```

#### Usage Examples

```cpp
// Basic FakeQuantize: 0..255 → 0..255 with u8 output
FakeQuantizeOnData fq(
    256ul,                    // quantizationLevels
    {},                       // constantShape (scalar)
    {0.f},                   // inputLowValues
    {255.f},                 // inputHighValues  
    {0.f},                   // outputLowValues
    {255.f},                 // outputHighValues
    ov::element::u8          // outputPrecision
);

// Per-channel quantization with specific shapes
FakeQuantizeOnData fqPerChannel(
    256ul,                    // quantizationLevels
    {1, 64, 1, 1},           // constantShape (per-channel for 64 channels)
    std::vector<float>(64, 0.f),      // inputLowValues (64 zeros)
    std::vector<float>(64, 255.f),    // inputHighValues (64 * 255)
    std::vector<float>(64, 0.f),      // outputLowValues 
    std::vector<float>(64, 255.f),    // outputHighValues
    ov::element::u8          // outputPrecision
);
```

## 4. Model Construction Implementation

This section explains where to find the code that constructs model parts based on LPT test structures. The framework provides specialized functions that convert the test structures described in previous sections into actual OpenVINO model nodes.

### Example: makeDequantization Function

The `makeDequantization()` function demonstrates how test structures are converted into model nodes. It takes a `DequantizationOperations` structure and builds the corresponding Convert→Subtract→Multiply operation chain in the model.

**Implementation Location**: [builders.cpp](../../../tests/ov_helpers/ov_lpt_models/src/common/builders.cpp)

### General Pattern

Following the same pattern as `makeDequantization()`, other model construction functions are organized in these directories:

- **Headers**: [ov_lpt_models/include/](../../../tests/ov_helpers/ov_lpt_models/include/ov_lpt_models/)
- **Implementation**: [ov_lpt_models/src/](../../../tests/ov_helpers/ov_lpt_models/src/)

These functions convert LPT test structures (like `DequantizationOperations`, `FakeQuantizeOnData`, etc.) into actual model nodes. Each operation-specific builder (Convolution, Add, Concat, etc.) uses these core construction functions to build complete test models from the test value structures.

---

*This completes the essential components needed to add test cases to the OpenVINO Low Precision Transformations framework. Use these patterns and helper functions to create comprehensive test coverage for your transformations.*

## See also

 * [OpenVINO™ Test Infrastructure](../../../tests/README.md)
 * [OpenVINO™ Low Precision Transformations documentation](../../../../docs/articles_en/documentation/openvino-extensibility/openvino-plugin-library/advanced-guides/low-precision-transformations.rst)
 * [OpenVINO™ README](../../../../README.md)
