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

import os
import ntpath


class Path:
    @staticmethod
    def get_model(model_file_path:str, addition:str = None, directory:str = None) -> str:
        if model_file_path is None:
            raise ValueError("model_file_path is None")

        file_name = ntpath.basename(model_file_path)
        model = os.path.splitext(file_name)
        if len(model) < 2:
            raise ValueError("model file name '{}' is not correct".format(file_name))
        if directory:
            return os.path.join(
                directory, 
                model[len(model) - 2] + (addition if addition else "") + ".xml")
        else:
            return os.path.join(
                os.path.dirname(model_file_path), 
                model[len(model) - 2] + (addition if addition else "") + ".xml")

    @staticmethod
    def get_weights(model_file_path:str, addition:str = None, directory:str = None) -> str:
        if model_file_path is None:
            raise ValueError("model_file_path is None")

        file_name = ntpath.basename(model_file_path)
        model = os.path.splitext(file_name)
        if len(model) < 2:
            raise ValueError("model file name '{}' is not correct".format(file_name))
        if directory:
            return os.path.join(
                directory, 
                model[len(model) - 2] + (addition if addition else "") + ".bin")
        else:
            return os.path.join(
                os.path.dirname(model_file_path), 
                model[len(model) - 2] + (addition if addition else "") + ".bin")

    @staticmethod
    def update_name(file_path: str, addition: str) -> str:
        file_name = ntpath.basename(file_path)
        parts = os.path.splitext(file_name)

        name = parts[0]
        extension = parts[-1] if len(parts) >= 2 else ""

        dir = os.path.dirname(file_path)
        return os.path.join(dir, name + addition + extension)
