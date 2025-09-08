# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Main entry-point to collect references for E2E tests.

Default run:
$ pytest collect_refs.py

Options[*]:
--modules       Paths to references
--env_conf      Path to environment config
--dry_run       Disable reference saving

[*] For more information see conftest.py
"""
# pylint:disable=invalid-name
import numpy as np
import logging as log
import os
from e2e_tests.common.parsers import pipeline_cfg_to_string
from e2e_tests.common.common.pipeline import Pipeline

pytest_plugins = ('e2e_tests.common.plugins.ref_collect.conftest', )


def save_reference(refs, path, use_torch_to_save):
    log.info("saving reference results to {path}".format(path=path))
    os.makedirs(os.path.dirname(path), mode=0o755, exist_ok=True)
    if use_torch_to_save:
        import torch
        torch.save(refs, path)
    else:
        np.savez(path, **refs)


def test_collect_reference(reference, dry_run):
    """Parameterized reference collection.

    :param reference: reference collection instance

    :param dry_run: dry-run flag. if True, disables saving reference result to
                    filesystem
    """
    for attr in ['pipeline', 'store_path']:
        if attr not in reference:
            raise ValueError(
                'obligatory attribute is missing: {attr}'.format(attr=attr))
    pipeline = Pipeline(reference['pipeline'])
    log.debug("Reference Pipeline:\n{}".format(pipeline_cfg_to_string(pipeline._config)))
    pipeline.run()
    refs = pipeline.fetch_results()
    if not dry_run:
        save_reference(refs, reference['store_path'], reference.get('use_torch_to_save', False))
        # Always save to `store_path_for_ref_save` (it points to share in automatics)
        if 'store_path_for_ref_save' in reference and reference['store_path'] != reference['store_path_for_ref_save']:
            save_reference(refs, reference['store_path_for_ref_save'], reference.get('use_torch_to_save', False))
    else:
        log.info("dry run option is used. reference results are not saved")




