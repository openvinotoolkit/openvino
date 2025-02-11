# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
import csv
import os
from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()

    accuracy_help = "Path to csv file with accuracy results"
    conformance_help = "Path to csv file with conformance failed models"
    model_help = "Path to file with full model list as .lst"
    device_help = "Target device"


    parser.add_argument("-a", "--accuracy", help=accuracy_help, required=True)
    parser.add_argument("-c", "--conformance", help=conformance_help, required=True)
    parser.add_argument("-m", "--model", help=model_help, required=True)
    parser.add_argument("-d", "--device", help=device_help, required=True)

    return parser.parse_args()

class Model:
    def __init__(self, model_name:str, model_framework:str, model_prc:str):
        self.__model_name = model_name
        self.__model_framework = model_framework
        self.__model_prc = model_prc
    
    def get_name(self):
        return self.__model_name

    def get_framework(self):
        return self.__model_framework

    def get_precision(self):
        return self.__model_prc

def path_to_model(model_path: os.path, prefix: str):
    frameworks = {'tf', 'tf2', 'caffe', 'onnx', 'paddle', 'kaldi'}
    precisions = {'FP16', 'FP32', 'INT8', 'INT1'}
    # remove share path + model.xml
    model_path = model_path.replace('\n', '')
    model, _ = os.path.split(re.sub(prefix, '', model_path))
    model, _ = os.path.split(model)
    model, _ = os.path.split(model)
    model_name, model_framework, model_prc = (None, None, None)
    for item in str(model).split(os.path.sep):
        if model_name == None:
            model_name = item
            continue
        if model_framework is None and item in frameworks:
            model_framework = item
            continue
        if item in precisions:
            if model_prc == None:
                model_prc = item
                continue
            model_prc += "-"
            model_prc += item
    return Model(model_name=model_name, model_framework=model_framework, model_prc=model_prc)

def process_model_list(model_list_file_path: os.path):
    if not os.path.isfile(model_list_file_path):
        raise Exception(f"Model filelist: {model_list_file_path} is not file!")
    models = set()
    with open(model_list_file_path, "r") as model_list_file:
        in_models = model_list_file.readlines()
        prefix = os.path.commonprefix(in_models)
        prefix += '(.*?)/'
        for line in in_models:
            models.add(path_to_model(line, prefix))
        model_list_file.close()
    return models

def convert_accuracy_res_to_bool(accuracy_status: str):
    conformance_like_status = accuracy_status
    if accuracy_status == "improvement":
        conformance_like_status = "passed"
    elif accuracy_status == "downgrade":
        conformance_like_status = "failed"

    return conformance_like_status

def process_accuracy(accuracy_res_file: os.path, target_device:str):
    if not os.path.isfile(accuracy_res_file):
        raise Exception(f"Model filelist: {accuracy_res_file} is not file!")
    # { model: status }
    results = dict()
    conformance_res = list()
    with open(accuracy_res_file, newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        model_name_row, model_name_row_idx = ("topology", None)
        framework_row, framework_row_idx = ("sourceframework", None)
        precision_row, precision_row_idx = ("precision", None)
        accuracy_row, accuracy_status_row_idx = ("metricstatus", None)
        device_row, device_row_idx = ("device", None)

        for row in csv_reader:
            if model_name_row_idx == None:
                for i in range(len(row)):
                    if row[i] == model_name_row:
                        model_name_row_idx = i
                    elif row[i] == framework_row:
                        framework_row_idx = i
                    elif row[i] == precision_row:
                        precision_row_idx = i
                    elif row[i] == accuracy_row:
                        accuracy_status_row_idx = i
                    elif row[i] == device_row:
                        device_row_idx = i
                continue
            model = Model(model_name=row[model_name_row_idx], model_framework=row[framework_row_idx], model_prc=row[precision_row_idx])
            if target_device in row[device_row_idx]:
                if model in results.keys():
                    old_status = convert_accuracy_res_to_bool(results[model])
                    new_status = convert_accuracy_res_to_bool(row[accuracy_status_row_idx])
                    if old_status != new_status and (new_status == "passed" or old_status == "not_found" or old_status == ""):
                        results[model] = new_status
                else:
                    results.update({model: convert_accuracy_res_to_bool(row[accuracy_status_row_idx])})
        csv_file.close()
    return results


def process_conformance(failed_models_path:os.path):
    if not os.path.isfile(failed_models_path):
        raise Exception(f"Conformance failed model csv: {failed_models_path} is not file!")
    model_paths = set()
    models = dict()
    with open(failed_models_path, newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for failed_model in csv_reader:
            model_paths.add(failed_model[0])
        csv_file.close()
    prefix = os.path.commonprefix(list(model_paths))
    prefix += '(.*?)/'
    for model in model_paths:
        models.update({path_to_model(model, prefix): "failed"})
    return models

def test_results_to_csv(model_set: set, accuracy_res: dict, conformance_res: dict):
    with open('test_results.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, dialect='excel')
        csv_writer.writerow(
            ["model_name",
             "framework",
             "precision",
             "accuracy_status",
             "conformance_status",
             "test_match"
             ])
        for model in model_set:
            accuracy_status = "not_found"
            conformance_status = "passed"
            for acc_model in accuracy_res.keys():
                if model.get_name() == acc_model.get_name() and \
                   model.get_framework() == acc_model.get_framework() and \
                   model.get_precision() == acc_model.get_precision():
                    accuracy_status = accuracy_res[acc_model]
                    break
            for conf_model in conformance_res.keys():
                if model.get_name() == conf_model.get_name() and \
                   model.get_framework() == conf_model.get_framework() and \
                   model.get_precision() == conf_model.get_precision():
                    conformance_status = conformance_res[conf_model]
                    break
            csv_writer.writerow([
                model.get_name(),
                model.get_framework(),
                model.get_precision(),
                accuracy_status,
                conformance_status,
                True if conformance_status == accuracy_status else False
                ])
        csv_file.close()

if __name__ == "__main__":
    args = parse_arguments()
    models = process_model_list(args.model)
    models_accuracy = process_accuracy(args.accuracy, args.device)
    models_conformance = process_conformance(args.conformance)
    test_results_to_csv(models, models_accuracy, models_conformance)
