import sys

from openvino.inference_engine import IECore


def param_to_string(metric):
    if isinstance(metric, (list, tuple)):
        return ", ".join([str(val) for val in metric])
    elif isinstance(metric, dict):
        str_param_repr = ""
        for k, v in metric.items():
            str_param_repr += "{}: {}\n".format(k, v)
        return str_param_repr
    else:
        return str(metric)


def main():
    ie = IECore()
    print("Available devices:")
    for device in ie.available_devices:
        print("\tDevice: {}".format(device))
        print("\tMetrics:")
        for metric in ie.get_metric(device, "SUPPORTED_METRICS"):
            try:
              metric_val = ie.get_metric(device, metric)
              print("\t\t{}: {}".format(metric, param_to_string(metric_val)))
            except TypeError:
              print("\t\t{}: UNSUPPORTED TYPE".format(metric))

        print("\n\tDefault values for device configuration keys:")
        for cfg in ie.get_metric(device, "SUPPORTED_CONFIG_KEYS"):
            try:
              cfg_val = ie.get_config(device, cfg)
              print("\t\t{}: {}".format(cfg, param_to_string(cfg_val)))
            except TypeError:
              print("\t\t{}: UNSUPPORTED TYPE".format(cfg))

if __name__ == '__main__':
    sys.exit(main() or 0)
