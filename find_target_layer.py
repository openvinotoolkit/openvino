import openvino as ov

def list_layers(model_path):
    core = ov.Core()
    model = core.read_model(model_path)
    
    print("Searching for FFN Down Projection inputs...")
    for op in model.get_ops():
        name = op.get_friendly_name()
        if "mlp" in name and "down_proj" in name and op.get_type_name() == "MatMul":
            print(f"Found Target: {name}")
            # The input to MatMul is usually index 0
            input_node = op.input_value(0).get_node()
            print(f"  Input Node: {input_node.get_friendly_name()} (Type: {input_node.get_type_name()})")
            # We only need one example
            break

if __name__ == "__main__":
    list_layers("model_ir/openvino_model.xml")
