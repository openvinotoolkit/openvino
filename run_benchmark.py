import sys
import time
from openvino.tools.benchmark.main import main

if __name__ == "__main__":
    # Simulate command line arguments
    sys.argv = [
        "benchmark_app",
        "-m", "model_ir/openvino_model.xml",
        "-d", "CPU",
        "-t", "15",
        "-api", "async",
        "-hint", "throughput",
        "-shape", "input_ids[1,128],attention_mask[1,128],position_ids[1,128]"
    ]
    main()
