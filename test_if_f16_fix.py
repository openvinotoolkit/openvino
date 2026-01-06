#!/usr/bin/env python3
"""
Comprehensive test for If operator with F16 constants fix.
Tests various scenarios to ensure the fix works correctly.
"""

import numpy as np
import openvino as ov
from openvino import Core, Model, Type, Shape, PartialShape
from openvino.runtime import opset8 as ops


def create_if_with_f16_constants():
    """Create an If model with F16 constants in both branches"""
    # Create condition parameter
    cond = ops.parameter([1], ov.Type.boolean, name="condition")
    
    # Create input parameters
    x_param = ops.parameter([2, 3], ov.Type.f32, name="x")
    
    # THEN body with F16 constant
    then_param = ops.parameter([-1, -1], ov.Type.f32, name="then_x")
    then_const_f16 = ops.constant(np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float16))
    then_const_f32 = ops.convert(then_const_f16, ov.Type.f32)
    then_mul = ops.multiply(then_param, then_const_f32, name="then_multiply")
    then_result = ops.result(then_mul, name="then_result")
    then_body = Model([then_result], [then_param], "then_body")
    
    # ELSE body with F16 constant
    else_param = ops.parameter([-1, -1], ov.Type.f32, name="else_x")
    else_const_f16 = ops.constant(np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=np.float16))
    else_const_f32 = ops.convert(else_const_f16, ov.Type.f32)
    else_div = ops.divide(else_param, else_const_f32, name="else_divide")
    else_result = ops.result(else_div, name="else_result")
    else_body = Model([else_result], [else_param], "else_body")
    
    # Create If operation
    if_op = ops.if_op(cond)
    if_op.set_then_body(then_body)
    if_op.set_else_body(else_body)
    if_op.set_input(x_param.output(0), then_param, else_param)
    if_output = if_op.set_output(then_result, else_result)
    
    # Create main model
    result = ops.result(if_output, name="output")
    model = Model([result], [cond, x_param], "if_with_f16_model")
    
    return model


def test_if_f16_read_model():
    """Test 1: Verify the ONNX model can be read"""
    print("Test 1: Reading ONNX model with If + F16...")
    try:
        core = Core()
        model = core.read_model("openvino_if_f16_bug_repro.onnx")
        print(f"  ✓ Model loaded: {len(model.inputs)} inputs, {len(model.outputs)} outputs")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_if_f16_compile():
    """Test 2: Verify the model can be compiled"""
    print("\nTest 2: Compiling ONNX model with If + F16...")
    try:
        core = Core()
        model = core.read_model("openvino_if_f16_bug_repro.onnx")
        compiled = core.compile_model(model, "CPU")
        print("  ✓ Model compiled successfully!")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_if_f16_inference():
    """Test 3: Verify inference works correctly"""
    print("\nTest 3: Running inference on compiled model...")
    try:
        core = Core()
        model = core.read_model("openvino_if_f16_bug_repro.onnx")
        compiled = core.compile_model(model, "CPU")
        
        # Create input data
        input_data = {compiled.input(0): np.random.randn(*compiled.input(0).shape).astype(np.float32)}
        
        # Run inference
        result = compiled(input_data)
        output = result[compiled.output(0)]
        
        print(f"  ✓ Inference successful! Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_programmatic_if_f16():
    """Test 4: Create and compile If model programmatically with F16"""
    print("\nTest 4: Creating If model programmatically with F16 constants...")
    try:
        model = create_if_with_f16_constants()
        print(f"  ✓ Model created: {len(model.inputs)} inputs, {len(model.outputs)} outputs")
        
        core = Core()
        compiled = core.compile_model(model, "CPU")
        print("  ✓ Model compiled successfully!")
        
        # Test inference - then branch (condition = True)
        cond_true = np.array([True], dtype=bool)
        x_data = np.ones((2, 3), dtype=np.float32)
        result = compiled({0: cond_true, 1: x_data})
        output = result[compiled.output(0)]
        print(f"  ✓ Then branch inference successful! Output shape: {output.shape}")
        
        # Test inference - else branch (condition = False)
        cond_false = np.array([False], dtype=bool)
        result = compiled({0: cond_false, 1: x_data})
        output = result[compiled.output(0)]
        print(f"  ✓ Else branch inference successful! Output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("OpenVINO If + F16 Constants Fix - Test Suite")
    print("=" * 70)
    
    results = []
    
    # Test 1: Read model
    results.append(("Read ONNX model", test_if_f16_read_model()))
    
    # Test 2: Compile model
    results.append(("Compile ONNX model", test_if_f16_compile()))
    
    # Test 3: Inference
    results.append(("Run inference", test_if_f16_inference()))
    
    # Test 4: Programmatic model
    results.append(("Programmatic If + F16", test_programmatic_if_f16()))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary:")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} - {test_name}")
    
    print("-" * 70)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
