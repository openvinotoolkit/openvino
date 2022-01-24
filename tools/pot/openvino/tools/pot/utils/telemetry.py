# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from ..engines.ac_engine import ACEngine
from ..engines.simplified_engine import SimplifiedEngine
from .logger import get_logger
from ..version import get_version as pot_version
try:
    import openvino_telemetry as tm
except ImportError:
    from . import telemetry_stub as tm

logger = get_logger(__name__)


def get_tid_telemetry():
    return 'UA-17808594-29'


def send_event(action,
               label,
               telemetry=tm.Telemetry(tid=get_tid_telemetry(), app_name='pot', app_version=pot_version())):
    try:
        telemetry.send_event('pot', action, label)
    except Exception as e: # pylint: disable=broad-except
        logger.info("Error occurred while trying to send telemetry. Details:" + str(e))


def send_configuration(algo_config, engine, interface='API'):
    try:
        telemetry = tm.Telemetry(tid=get_tid_telemetry(), app_name='pot', app_version=pot_version)

        target_device = ','.join(set(algorithm['params'].get('target_device', 'ANY') for algorithm in algo_config))
        algorithms = {f'algorithm_{i}': algorithm['name'] for i, algorithm in enumerate(algo_config)}
        get_subset_size = lambda i: min(algo_config[i]['params'].get('stat_subset_size', len(engine.data_loader)),
                                        len(engine.data_loader))
        stat_subset_size = ','.join([str(get_subset_size(i)) for i, _ in enumerate(algo_config)])
        model_type = algo_config[0]['params'].get('model_type', None) if len(algo_config) > 0 else str(None)
        engine_type = 'simplified' if isinstance(engine, SimplifiedEngine) else \
                      'accuracy_checker' if isinstance(engine, ACEngine) else 'engine'

        for algo in algo_config:
            if algo['name'] == 'AccuracyAwareQuantization':
                drop_type_aa = algo['params'].get('drop_type', 'absolute')
                maximal_drop_aa = algo['params'].get('maximal_drop', None)
                tune_hyperparams = algo['params'].get('tune_hyperparams', False)
                send_event('drop_type_aa', drop_type_aa, telemetry)
                send_event('maximal_drop_aa', str(maximal_drop_aa), telemetry)
                send_event('tune_hyperparams', tune_hyperparams, telemetry)

        send_event('target_device', target_device, telemetry)
        send_event('algorithms', str(algorithms), telemetry)
        send_event('stat_subset_size', stat_subset_size, telemetry)
        send_event('model_type', str(model_type), telemetry)
        send_event('engine_type', engine_type, telemetry)
        send_event('interface', interface, telemetry)
    except Exception as e: # pylint: disable=broad-except
        logger.info("Error occurred while trying to send telemetry. Details:" + str(e))


def start_session_telemetry():
    try:
        telemetry = tm.Telemetry(tid=get_tid_telemetry(), app_name='pot', app_version=pot_version)
        telemetry.start_session('pot')
        return telemetry
    except Exception as e: # pylint: disable=broad-except
        logger.info("Error occurred while trying to send telemetry. Details:" + str(e))
        return None


def end_session_telemetry(telemetry):
    try:
        telemetry.end_session('pot')
        telemetry.force_shutdown(1.0)
    except Exception as e: # pylint: disable=broad-except
        logger.info("Error occurred while trying to send telemetry. Details:" + str(e))
