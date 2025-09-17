#!/usr/bin/env python

#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import json
import jsonschema
from pathlib import Path
from functools import cached_property

class JSONSchemaValidator(dict):
    @staticmethod
    def load_from_file(schema_name):
        schema_name += ".json"
        file_schema = Path(__file__).parent.parent / "schema" / schema_name
        with file_schema.open() as f:
            loaded_content = json.load(f)

        return loaded_content

    @staticmethod
    def is_valid(json_schema, input_json):
        valid = True
        try:
            jsonschema.validate(instance=input_json, schema=json_schema)
        except jsonschema.exceptions.ValidationError as ex:
            valid = False
            pass
        return valid

    def __init__(self, json_schema, input_json, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            jsonschema.validate(instance=input_json, schema=json_schema)
        except jsonschema.exceptions.ValidationError as ex:
            raise RuntimeError(f"Provided JSON data: {input_json}\nis doesn't met the schema:\n{json_schema}.\nValidation details: {str(ex)}") from None

        self.update(input_json)
