# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

schema_dict = {
    "definitions": {},
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Root",
    "type": "array",
    "default": [],
    "items": {
        "$id": "#root/items",
        "title": "Items",
        "type": "object",
        "required": [
            "id",
            "match_kind"
        ],
        "properties": {
            "custom_attributes": {
                "$id": "#root/items/custom_attributes",
                "title": "Custom_attributes",
                "type": "object",
                "properties": {
                }
            },
            "id": {
                "$id": "#root/items/id",
                "title": "Id",
                "type": "string",
                "pattern": "^.*$",
                "minLength": 1
            },
            "inputs": {
                "$id": "#root/items/inputs",
                "title": "Inputs",
                "type": "array",
                "default": [],
                "items": {
                    "$id": "#root/items/inputs/items",
                    "title": "Items",
                    "type": "array",
                    "default": [],
                    "items": {
                        "$id": "#root/items/inputs/items/items",
                        "title": "Items",
                        "type": "object",
                        "properties": {
                            "node": {
                                "$id": "#root/items/inputs/items/items/node",
                                "title": "Node",
                                "type": "string",
                                "default": "",
                                "pattern": "^.*$"
                            },
                            "port": {
                                "$id": "#root/items/inputs/items/items/port",
                                "title": "Port",
                                "type": "integer",
                                "default": 0
                            }
                        },
                        "required": ["node", "port"]
                    }

                }
            },
            "instances": {
                "$id": "#root/items/instances",
                "title": "Instances",
                "type": ["array", "object"],
                "items": {
                    "$id": "#root/items/instances/items",
                    "title": "Items",
                    "type": "string",
                    "default": "",
                    "pattern": "^.*$"
                }
            },
            "match_kind": {
                "$id": "#root/items/match_kind",
                "title": "Match_kind",
                "type": "string",
                "enum": ["points", "scope", "general"],
                "default": "points",
                "pattern": "^.*$"
            },
            "outputs": {
                "$id": "#root/items/outputs",
                "title": "Outputs",
                "type": "array",
                "default": [],
                "items": {
                    "$id": "#root/items/outputs/items",
                    "title": "Items",
                    "type": "object",
                    "properties": {
                        "node": {
                            "$id": "#root/items/outputs/items/node",
                            "title": "Node",
                            "type": "string",
                            "default": "",
                            "pattern": "^.*$"
                        },
                        "port": {
                            "$id": "#root/items/outputs/items/port",
                            "title": "Port",
                            "type": "integer",
                            "default": 0
                        }
                    },
                    "required": ["node", "port"]
                }

            },
            "include_inputs_to_sub_graph": {
                "$id": "#root/items/include_inputs_to_sub_graph",
                "title": "Include_inputs_to_sub_graph",
                "type": "boolean",
                "default": False
            },
            "include_outputs_to_sub_graph": {
                "$id": "#root/items/include_outputs_to_sub_graph",
                "title": "Include_outputs_to_sub_graph",
                "type": "boolean",
                "default": False
            }
        }
    }
}
