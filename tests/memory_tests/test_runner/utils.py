# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility module."""

import logging
import sys

from pymongo import MongoClient

# constants
REFS_FACTOR = 1.2  # 120%
TIMELINE_SIMILARITY = ('model', 'device', 'test_exe', 'os', 'cpu_info', 'target_branch')


def _transpose_dicts(items, template=None):
    """ Build dictionary of arrays from array of dictionaries
    Example:
    > in = [{'a':1, 'b':3}, {'a':2}]
    > _transpose_dicts(in, template=in[0])
    {'a':[1,2], 'b':[3, None]}
    """
    result = {}
    if not items:
        return result
    if not template:
        template = items[0]
    for key, template_val in template.items():
        if isinstance(template_val, dict):
            result[key] = _transpose_dicts(
                [item[key] for item in items if key in item], template_val)
        else:
            result[key] = [item.get(key, None) for item in items]
    return result


def query_memory_timeline(records, db_url, db_name, db_collection, max_items=20, similarity=TIMELINE_SIMILARITY):
    """ Query database for similar memory items committed previously
    """

    def timeline_key(item):
        """ Defines order for timeline report entries
        """
        order = 0
        for step_name, _ in item['results'].items():
            if len(item['results'][step_name]['vmhwm']) <= 1:
                return 1
            order = item['results'][step_name]['vmhwm']["avg"][-1] - item['results'][step_name]['vmhwm']["avg"][-2] + \
                    item['results'][step_name]['vmrss']["avg"][-1] - item['results'][step_name]['vmrss']["avg"][-2]
            if not item['status']:
                # ensure failed cases are always on top
                order += sys.maxsize / 2
        return order

    client = MongoClient(db_url)
    collection = client[db_name][db_collection]
    result = []
    for record in records:
        items = []
        try:
            query = dict((key, record[key]) for key in similarity)
            query['commit_date'] = {'$lt': record['commit_date']}
            pipeline = [
                {'$match': query},
                {'$addFields': {
                    'commit_date': {'$dateFromString': {'dateString': '$commit_date'}}}},
                {'$sort': {'commit_date': -1}},
                {'$limit': max_items},
                {'$sort': {'commit_date': 1}},
            ]
            items += list(collection.aggregate(pipeline))
        except KeyError:
            pass  # keep only the record if timeline failed to generate
        items += [record]
        for item in items:
            item["status"] = {"passed": True, "failed": False, "not_finished": False}[item["status"]]
        timeline = _transpose_dicts(items, template=record)
        result += [timeline]

    result.sort(key=timeline_key, reverse=True)
    return result


def compare_with_references(aggr_stats: dict, reference: dict):
    """Compare values with provided reference"""

    vm_metrics_to_compare = {"vmrss", "vmhwm"}
    stat_metrics_to_compare = {"avg"}
    status = 0

    for step_name, vm_records in reference.items():
        for vm_metric, stat_metrics in vm_records.items():
            if vm_metric not in vm_metrics_to_compare:
                continue
            for stat_metric_name, reference_val in stat_metrics.items():
                if stat_metric_name not in stat_metrics_to_compare:
                    continue
                if aggr_stats[step_name][vm_metric][stat_metric_name] > reference_val * REFS_FACTOR:
                    logging.error(f"Comparison failed for '{step_name}' step for '{vm_metric}' for"
                                  f" '{stat_metric_name}' metric. Reference: {reference_val}."
                                  f" Current values: {aggr_stats[step_name][vm_metric][stat_metric_name]}")
                    status = 1
                else:
                    logging.info(f"Comparison passed for '{step_name}' step for '{vm_metric}' for"
                                 f" '{stat_metric_name}' metric. Reference: {reference_val}."
                                 f" Current values: {aggr_stats[step_name][vm_metric][stat_metric_name]}")
    return status
