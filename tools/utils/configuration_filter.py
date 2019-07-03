"""
Copyright (C) 2018-2019 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import copy
import os


class ConfigurationFilter:
    @staticmethod
    def filter(configuration, filter_metric_name: str, filter_metric_type: str, logger = None):
        updated_configuration = copy.deepcopy(configuration)
        if 'models' not in updated_configuration or len(updated_configuration['models']) == 0:
            raise ValueError("'models' key is absent in configuration")

        updated_configuration['models'] = [model for model in updated_configuration['models'] if 'launchers' in model and model['launchers']]
        if len(updated_configuration['models']) > 1:
            raise ValueError("too many models")

        if not updated_configuration['models']:
            raise ValueError("there are no models")

        model = updated_configuration['models'][0]
        if 'datasets' not in model or len(model['datasets']) == 0:
            raise ValueError("'datasets' key is absent in models")

        if len(model['datasets']) > 1:
            raise ValueError("too many datasets in model")

        dataset = model['datasets'][0]
        if filter_metric_name:
            dataset['metrics'] = [i for i in dataset['metrics'] if i['name'] == filter_metric_name]

        if filter_metric_type:
            dataset['metrics'] = [i for i in dataset['metrics'] if i['type'] == filter_metric_type]

        if 'metrics' not in dataset or len(dataset['metrics']) == 0:
            raise ValueError("can not find appropriate metric in dataset{}{}".format(
                ", filter_metric_name='{}'".format(filter_metric_name) if filter_metric_name else "",
                ", filter_metric_type='{}'".format(filter_metric_type) if filter_metric_type else ""))

        if filter_metric_name is None and filter_metric_type is None and len(dataset['metrics']) > 1:
            dataset['metrics'] = [dataset['metrics'][0]]
            if logger:
                logger.warn("too many metrics without filters, first metric '{}' is used".format(str(dataset['metrics'][0])))

        if len(dataset['metrics']) > 1:
            raise ValueError("too many metrics in datasets")

        metric = dataset['metrics'][0]
        if 'presenter' in metric and metric['presenter'] != 'return_value':
            original_presenter = metric['presenter']
            metric['presenter'] = 'return_value'
            if logger:
                logger.warn("presenter was changed from '{}' to '{}'".format(original_presenter, metric['presenter']))
        else:
            metric['presenter'] = 'return_value'
            if logger:
                logger.warn("presenter was set to '{}'".format(metric['presenter']))

        return updated_configuration

