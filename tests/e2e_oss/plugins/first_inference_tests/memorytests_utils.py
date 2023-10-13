import copy
import logging as log
import os
import statistics
import tempfile
import yaml
from .common_utils import get_executable_cmd, deviation
from ...common_utils.test_utils import shell


def parse_stats(stats: dict, res: dict):
    """Parse statistics to dict"""
    for k, v in stats.items():
        if k not in res.keys():
            res.update({k: {}})
        if isinstance(v, list):
            for element in v:
                for metric, value in element.items():
                    res[k].update({metric: [value]})


def append_stats(stats: dict, parsed_dict: dict):
    if not stats:
        return copy.deepcopy(parsed_dict)
    for step_name, vm_values in parsed_dict.items():
        for vm_metric, vm_value in vm_values.items():
            stats[step_name][vm_metric].extend(vm_value)
    return stats


def aggregate_stats(stats: dict):
    aggregated_stats = {step_name: {vm_metric: {"avg": statistics.mean(vm_values_list),
                                    "stdev": statistics.stdev(vm_values_list) if len(vm_values_list) > 1 else 0}
                                    for vm_metric, vm_values_list in vm_values[0].items()
                                    if vm_metric in ("vmrss", "vmhwm")}
                        for step_name, vm_values in stats.items()}
    return aggregated_stats


def run_memorytest(args: dict):
    """Run provided executable several times and aggregate collected statistics"""
    cmd_common = get_executable_cmd(args)
    # Run executable and collect statistics
    stats = {}
    for run_iter in range(args["niter"]):
        tmp_stats_path = tempfile.NamedTemporaryFile().name
        retcode, _, stderr = shell(cmd_common + ["-s", str(tmp_stats_path)], log=log)
        if retcode != 0:
            log.error(f"Run of executable '{args['executable']}' failed with return code '{retcode}'. "
                      f"Error: {stderr}\n"
                      f"Statistics aggregation is skipped.")
            return retcode, stderr, {}, {}

        # Read raw statistics
        with open(tmp_stats_path, "r") as file:
            raw_data = list(yaml.load_all(file, Loader=yaml.SafeLoader))

        os.unlink(tmp_stats_path)

        # Parse raw data
        flatten_data = {}
        parse_stats(raw_data[0], flatten_data)

        log.debug(f"Statistics after run of executable #{run_iter}: {flatten_data}")

        # Combine statistics from several runs
        stats.update((step_name, stats.get(step_name, []) + [memory_metric])
                     for step_name, memory_metric in flatten_data.items())

    # Aggregate results
    aggregated_stats = aggregate_stats(stats)
    log.debug(f"Aggregated statistics after full run: {aggregated_stats}")

    return aggregated_stats


def get_compared_memory_results(static_res, dynamic_res):
    """Returns the difference between static and dynamic results as percentage"""
    if isinstance(static_res, tuple):
        raise ValueError(static_res[1])
    if isinstance(dynamic_res, tuple):
        raise ValueError(dynamic_res[1])
    if "reshape" in dynamic_res:
        del dynamic_res["reshape"]
    compared_res = {}
    for static_step_name, static_avg in static_res.items():
        compared_res.update({static_step_name: {}})
        for memory_type, memory_type_stats in static_avg.items():
            dynamic_res_avg = dynamic_res[static_step_name][memory_type]['avg']
            static_res_avg = memory_type_stats['avg']
            diff_prc = deviation(dynamic_res_avg, static_res_avg)
            compared_res[static_step_name].update(
                {memory_type: {"dynamic": str(dynamic_res_avg), "static": str(static_res_avg), "ratio": f'{diff_prc}'}})
    return compared_res


def get_compared_with_refs_results(ref_res, cur_res):
    """Returns the difference between reference and current memory results"""
    ref_compared_res = {}
    if isinstance(ref_res, dict):
        for metric_ref_name, metric_ref_value in ref_res.items():
            ref_compared_res.update({metric_ref_name: {}})
            for memory_type, memory_type_stats in metric_ref_value.items():
                reference_value = float(memory_type_stats['ratio'])
                current_value = float(cur_res[metric_ref_name][memory_type]["ratio"])
                ratio = deviation(reference_value, current_value)
                ref_compared_res[metric_ref_name].update({memory_type: {
                    "ratio": ratio,
                    "reference_value": reference_value,
                    "current_value": current_value
                }})
    else:
        ref_compared_res["ErrorOutput"] = ref_res
    return ref_compared_res
