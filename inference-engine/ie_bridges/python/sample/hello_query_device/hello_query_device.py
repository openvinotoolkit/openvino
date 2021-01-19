import sys

from openvino.inference_engine import IECore


def param_to_string(metric):
    if isinstance(metric, (list, tuple)):
        return ", ".join([str(val) for val in metric])
    elif isinstance(metric, dict):
        str_param_repr = ""
        for k, v in metric.items():
            str_param_repr += f"{k}: {v}\n"
        return str_param_repr
    else:
        return str(metric)


def main():
    ie = IECore()
    print("Available devices:")
    for device in ie.available_devices:
        print(f"\tDevice: {device}")
        print("\tMetrics:")
        for metric in ie.get_metric(device, "SUPPORTED_METRICS"):
            try:
              metric_val = ie.get_metric(device, metric)
              print(f"\t\t{metric}: {param_to_string(metric_val)}")
            except TypeError:
              print(f"\t\t{metric}: UNSUPPORTED TYPE")

        print("\n\tDefault values for device configuration keys:")
        for cfg in ie.get_metric(device, "SUPPORTED_CONFIG_KEYS"):
            try:
              cfg_val = ie.get_config(device, cfg)
              print(f"\t\t{cfg}: {param_to_string(cfg_val)}")
            except TypeError:
              print(f"\t\t{cfg}: UNSUPPORTED TYPE")

if __name__ == '__main__':
    sys.exit(main() or 0)
