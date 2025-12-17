import openvino as ov
import sys
import os

# Add extension path
if os.name == 'nt':
    openvino_bin = r"C:\Users\ssdaj\openvino\bin\intel64\Release"
    if os.path.exists(openvino_bin):
        os.add_dll_directory(openvino_bin)

def inspect_tssn(model_path):
    core = ov.Core()
    # Load extension
    ext_path = r"src/custom_ops/build/Release/openvino_tssn_extension.dll"
    if os.path.exists(ext_path):
        core.add_extension(ext_path)
    else:
        print(f"Extension not found at {ext_path}")

    try:
        model = core.read_model(model_path)
    except Exception as e:
        print(f"Error reading model: {e}")
        return

    print(f"Inspecting model: {model.friendly_name}")
    
    target_add = "Add_223726"
    found_add = False

    for op in model.get_ops():
        if op.friendly_name == target_add:
            found_add = True
            print(f"\nFOUND TARGET ADD: {op.friendly_name}")
            for i, inp in enumerate(op.inputs()):
                ps = inp.get_partial_shape()
                src_node = inp.get_source_output().get_node()
                print(f"  Input {i}: {ps} from {src_node.get_type_name()} ({src_node.friendly_name})")
        
        if op.get_type_name() == "CompositeTSSN" or ("down_proj" in op.friendly_name and op.get_type_name() == "MatMul"):
            print(f"\n{op.get_type_name()}: {op.friendly_name}")
            for i, inp in enumerate(op.inputs()):
                ps = inp.get_partial_shape()
                print(f"  Input {i}: {ps}")
            for i, out in enumerate(op.outputs()):
                ps = out.get_partial_shape()
                print(f"  Output {i}: {ps}")
                
                # Check consumers
                for target_input in out.get_target_inputs():
                    consumer = target_input.get_node()
                    print(f"    -> Consumed by: {consumer.get_type_name()} ({consumer.friendly_name})")
                    
                    # Trace deeper
                    for out2 in consumer.outputs():
                        for target_input2 in out2.get_target_inputs():
                            consumer_l2 = target_input2.get_node()
                            print(f"      -> L2 Consumer: {consumer_l2.get_type_name()} ({consumer_l2.friendly_name})")
                            
                            if consumer_l2.get_type_name() == "Add":
                                print(f"        Add Inputs:")
                                for j, add_inp in enumerate(consumer_l2.inputs()):
                                    add_ps = add_inp.get_partial_shape()
                                    print(f"          Input {j}: {add_ps} from {add_inp.get_source_output().get_node().friendly_name}")
                            
                            # Trace L3
                            for out3 in consumer_l2.outputs():
                                for target_input3 in out3.get_target_inputs():
                                    consumer_l3 = target_input3.get_node()
                                    # print(f"        -> L3 Consumer: {consumer_l3.get_type_name()} ({consumer_l3.friendly_name})")
                                    if consumer_l3.get_type_name() == "Add":
                                        print(f"          L3 Add ({consumer_l3.friendly_name}) Inputs:")
                                        for j, add_inp in enumerate(consumer_l3.inputs()):
                                            add_ps = add_inp.get_partial_shape()
                                            print(f"            Input {j}: {add_ps} from {add_inp.get_source_output().get_node().friendly_name}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_tssn(sys.argv[1])
    else:
        print("Usage: python inspect_tssn_shapes.py <model_xml>")
