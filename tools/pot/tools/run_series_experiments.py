# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from argparse import ArgumentParser
from copy import deepcopy
from pprint import pformat

#TODO: avoid this trick
sys.path.insert(0, Path(__file__).resolve().parent.parent.as_posix()) # pylint: disable=C0413

from openvino.tools.pot.utils.timestamp import get_timestamp, get_timestamp_short
from openvino.tools.pot.utils.logger import init_logger, get_logger
from openvino.tools.pot.utils.config_reader import read_config_from_file

# Attention: this is the major function `main` from main.py
from openvino.tools.pot.app.run import app as main_in_experiment


logger = get_logger(__name__)


def _parse_experiment_params(params):
    assert isinstance(params, dict), 'Experiment params in config should be a dict'
    params = deepcopy(params)
    cur_lens = []
    broadcast_keys = []
    for k, v in params.items():
        if isinstance(v, (list, tuple)) and len(v) > 1:
            cur_lens.append((k, len(v)))
        else:
            broadcast_keys.append(k)

    if cur_lens:
        first_key, first_len = cur_lens[0]
        num_experiments = first_len
    else:
        first_key = None
        num_experiments = 1

    # check lens
    for k, n in cur_lens:
        if n != num_experiments:
            raise RuntimeError('The length of parameter {} in config file is greater than 1 and '
                               'differs from the length of the parameter {} ({} vs {}). '
                               'Note that all parameters of the experiments that are lists with length '
                               'greater than 1 MUST have THE SAME length, whereas other will be BROADCASTED '
                               'to have this length.'.format(k, first_key, n, num_experiments))

    # broadcasting
    for k in broadcast_keys:
        v = params[k]
        if not isinstance(v, (list, tuple)):
            params[k] = [v,] * num_experiments
        else:
            assert len(v) == 1
            params[k] = [v[0],] * num_experiments

    for v in params.values():
        assert isinstance(v, (list, tuple)) and len(v) == num_experiments

    return params, num_experiments


def _prepare_experiment_folders(series_dir, index, etimestamp):
    series_dir = Path(series_dir)
    series_dir.mkdir(parents=True, exist_ok=True)
    series_dir = series_dir.resolve()

    def _make_subfolder(folder, name):
        sub_path = folder / name
        sub_path.mkdir(exist_ok=True)
        return sub_path.resolve()

    cur_exp_dir = _make_subfolder(series_dir, 'exp_' + etimestamp + '_{:04}'.format(index))
    out_exp_dir = _make_subfolder(cur_exp_dir, 'compressed_models')
    log_exp_dir = _make_subfolder(cur_exp_dir, 'logs')

    exp_config = cur_exp_dir / ('experiment_config_' + etimestamp + '.yml')
    exp_config = exp_config.absolute()
    return cur_exp_dir, out_exp_dir, log_exp_dir, exp_config


def _create_exp_config_from_template(experiment_template_path, cur_params, exp_config):
    template_str = experiment_template_path.read_text()

    assert isinstance(cur_params, dict)
    config_str = template_str.format(**cur_params)
    exp_config.write_text(config_str)


def _run_pipeline_from_config(exp_config, out_exp_dir, dry_run):
    # Attention: these lines execute the function `main` from `main.py`
    # Be careful.
    cmd = ['--config', str(exp_config),
           '--output-dir', str(out_exp_dir)]

    logger.info('Parameters for main in experiment =\n{}'.format(cmd))

    if not dry_run:
        main_in_experiment(cmd)


def _run_experiment(experiment_template_path, cur_params,
                    index, series_dir, stimestamp, root_dir,
                    dry_run):
    etimestamp = get_timestamp_short()

    cur_exp_dir, out_exp_dir, log_exp_dir, exp_config = \
            _prepare_experiment_folders(series_dir, index, etimestamp)

    additional_params = {
        'INDEX': index, #index of experiment in the series
        'STIMESTAMP': stimestamp, #timestamp of the current series of experiments
        'ETIMESTAMP': etimestamp, #timestamp of the current experiment
        'CUR_DIR': cur_exp_dir,  #folder of the current experiment
        'OUT_DIR': out_exp_dir, #folder where compressed models will be stored in the cur. experiment
        'LOG_DIR': log_exp_dir, #log folder of the current experiment
        'EXP_CONFIG': exp_config, #path to config file of the current experiment
        'SERIES_DIR': series_dir, #folder of the current series of experiments
        'ROOT_DIR': root_dir, #root folder, where the folders for series of experiments are created
        }
    for k in additional_params:
        if k in cur_params:
            raise RuntimeError('ERROR: The reserved parameter name "{}" is used '
                               'in experimets config file'.format(k))

    cur_params.update(additional_params)
    logger.debug('after updating with additional_params cur_params=\n{}'.format(pformat(cur_params)))
    _create_exp_config_from_template(experiment_template_path, cur_params, exp_config)
    logger.debug('Created config file for the current experiment\n{}'.format(exp_config))

    _run_pipeline_from_config(exp_config, out_exp_dir, dry_run=dry_run)


def _run_experiments(experiment_template_path, experiment_params, num_experiments,
                     series_dir, stimestamp, root_dir,
                     dry_run):
    assert all(isinstance(v, (list, tuple)) and len(v) == num_experiments
               for v in experiment_params.values()), 'Wrong experiment parameters -- inner error'

    for index in range(num_experiments):
        logger.info('Begin experiment {}'.format(index))
        cur_params = {}
        for k, v in experiment_params.items():
            cur_params[k] = v[index]

        logger.debug('Experiment parameters =\n{}'.format(pformat(cur_params)))

        _run_experiment(experiment_template_path=experiment_template_path,
                        cur_params=cur_params,
                        index=index,
                        series_dir=series_dir,
                        stimestamp=stimestamp,
                        root_dir=root_dir,
                        dry_run=dry_run)
        logger.info('End experiment {}'.format(index))
        logger.info('=' * 80)


def main():
    stimestamp = get_timestamp()
    parser = ArgumentParser(description='Tool for making series of experiments', allow_abbrev=False)
    parser.add_argument('-c', '--config', required=True, help='Path to a config file')
    parser.add_argument('-o', '--output-dir', default='.',
                        help='Path to a folder to store the results, default is ".".')

    log_levels = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']
    parser.add_argument('--log-level', type=str, default='INFO', choices=log_levels,
                        help='Log level to print: {}'.format(log_levels))

    parser.add_argument('--dry-run', action='store_true',
                        help='If everything excepth actual run of experiment should be done')

    args = parser.parse_args()
    dry_run = args.dry_run

    init_logger(level=args.log_level, stream=sys.stdout)

    config_path = Path(args.config)
    root_dir = args.output_dir
    root_dir = Path(root_dir)
    series_dir = root_dir / ('series_experiments_' + stimestamp)
    series_dir = series_dir.absolute()

    config = read_config_from_file(config_path)

    experiment_template_path = Path(config['experiment_template']).absolute()
    if not experiment_template_path.exists():
        raise RuntimeError('Cannot find experiment template "{}"'.format(experiment_template_path))

    experiment_params, num_experiments = _parse_experiment_params(config['params'])

    _run_experiments(experiment_template_path=experiment_template_path,
                     experiment_params=experiment_params,
                     num_experiments=num_experiments,
                     series_dir=series_dir,
                     stimestamp=stimestamp,
                     root_dir=root_dir,
                     dry_run=dry_run)


if __name__ == '__main__':
    main()
