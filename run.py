import openvino as ov
print("OpenVINO version:", ov.__version__)
from collections import defaultdict

model_path = "/home/estepyre/repos/openvino/facenet.pb"


def report_all_ops(model):
    ops = defaultdict(int)
    sub_ops = defaultdict(int)
    for op in model.get_ordered_ops():
        ops[op.get_type_name()] += 1
        if op.get_type_name() == "If":
            for subgraph in [op.get_then_body(), op.get_else_body()]:
                for node in subgraph.get_ordered_ops():
                    sub_ops[node.get_type_name()] += 1

    def print_dict(data, title):
        print(f"{title}:")
        for name, count in sorted(data.items()):
            print(name, count)

    print_dict(ops, "Graph ops")
    print()
    print_dict(sub_ops, "Subgraph ops")

if __name__ == "__main__":
    print("Convert Model")
    model = ov.convert_model(model_path, input=(1,160,160,3))
    report_all_ops(model)

    print("Reshape")
    model.reshape({'batch_size:0': [1, 160, 160, 3]})
