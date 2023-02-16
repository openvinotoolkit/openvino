# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from shutil import copyfile
from collections import namedtuple, OrderedDict
from pathlib import Path
from addict import Dict

from .open_model_zoo import download as omz_model_download
from .open_model_zoo import convert as omz_model_convert
from .custom_models import convert_custom as custom_model_convert
from .open_model_zoo import get_models_list as get_omz_models_list
from .custom_models import get_models_list as get_custom_models_list
from .open_model_zoo import DOWNLOAD_PATH

Model = namedtuple('Model', ['model_name', 'framework', 'model_params'])


class ModelStore:
    def __init__(self):
        self.models = []
        # load model description to self.models
        self._load_models_description()

    def get(self, name, framework, tmp_path, model_precision='FP32', custom_mo_config=None):
        for model in self.models:
            if framework != model.framework:
                continue
            if not (name == model.name or (name + '-' + framework) == model.name):
                continue

            model.model_params.output_dir = tmp_path.as_posix()
            model.precision = model_precision
            if not model.downloaded:
                if model.source != 'omz':
                    raise RuntimeError(
                        'Couldn\'t load model {} from the framework {}'.format(model.name, model.framework))
                assert omz_model_download(model) == 0,\
                    'Can not download model: {}'.format(model.name)
                convert_value = omz_model_convert(model, custom_mo_config)
                assert convert_value == 0, 'Can not convert model: {}'.format(model.name)
                model_path = tmp_path.joinpath(
                    model.subdirectory.as_posix(), model.precision, model.name)
                if not os.path.isfile(model_path.as_posix() + '.xml'):
                    omz_path = Path(DOWNLOAD_PATH)
                    model_omz_path = omz_path.joinpath(
                        model.subdirectory.as_posix(), model.precision, model.name)
                    source_xml_path = model_omz_path.as_posix() + '.xml'
                    model_path.mkdir(parents=True)
                    copyfile(source_xml_path, model_path.as_posix() + '.xml')
                    source_bin_path = model_omz_path.as_posix() + '.bin'
                    copyfile(source_bin_path, model_path.as_posix() + '.bin')
            else:
                model_path = tmp_path.joinpath(model.name)
                if model.framework == 'dldt':
                    source_xml_path = model.mo_args.model
                    copyfile(source_xml_path, model_path.as_posix() + '.xml')
                    source_bin_path = model.mo_args.weights
                    if source_bin_path and os.path.isfile(source_bin_path):
                        copyfile(source_bin_path, model_path.as_posix() + '.bin')
                else:
                    assert custom_model_convert(model) == 0,\
                        'Can not convert model: {}'.format(model.name)
            model.model_params.model = model_path.as_posix() + '.xml'
            model.model_params.weights = model_path.as_posix() + '.bin'
            model.model_params.model_name = model.name
            return Model(model.name, model.framework, Dict(model.model_params))
        return None

    def _load_models_description(self):
        sources = OrderedDict([('custom', get_custom_models_list),
                               ('omz', get_omz_models_list)])

        for source, fn in sources.items():
            for t in fn():
                self.models.append(
                    Dict({'name': t.name,
                          'framework': t.framework,
                          'mo_args': t.mo_args,
                          'model_params': {},
                          'downloaded': source != 'omz',
                          'source': source,
                          'subdirectory': t.subdirectory
                          }))

    def get_cascade(self, name, framework, tmp_path, cascade_props, model_precision='FP32'):
        cascade_props = Dict(cascade_props)
        model_name_ = cascade_props.main_model
        cascade = {m_name: self.get(m_name, framework, tmp_path, model_precision)
                   for m_name in cascade_props.model_names}
        model = cascade[model_name_]
        model.model_params.model_name = name
        model.model_params.update(
            Dict({'cascade': [
                {
                    'name': token,
                    'model': cascade[name_].model_params.model,
                    'weights': cascade[name_].model_params.weights
                }
                for token, name_ in zip(cascade_props.model_tokens, cascade_props.model_names)]})
        )

        return model
