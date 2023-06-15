# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


def get_convert_model_help_specifics():
    from openvino.runtime.utils.cli_parser import CanonicalizeTransformationPathCheckExistenceAction, \
        CanonicalizePathCheckExistenceAction, CanonicalizeExtensionsPathCheckExistenceAction, \
        CanonicalizePathCheckExistenceIfNeededAction, readable_file_or_dir, readable_dirs_or_files_or_empty, \
        check_positive
    from openvino.runtime.utils.version import VersionChecker

    return {
        'input_model':
            {'action': CanonicalizePathCheckExistenceAction,
             'type': readable_file_or_dir},
        'extensions':
            {'action': CanonicalizeExtensionsPathCheckExistenceAction,
             'type': readable_dirs_or_files_or_empty},
        'transformations_config':
            {'action': CanonicalizeTransformationPathCheckExistenceAction},
        'counts':
            {'action': CanonicalizePathCheckExistenceIfNeededAction},
        'version':
            {'action': 'version',
             'version': 'Version of Conversion API is: {}'.format(VersionChecker().get_ie_version())},
        'scale':
            {'type': float,
             'aliases': {'-s'}},
        'batch':
            {'type': check_positive,
             'aliases': {'-b'}},
        'input_proto':
            {'aliases': {'-d'}},
        'log_level':
            {'choices': ['CRITICAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET']}
    }


# TODO: remove this when internal converting of params to string is removed
def get_to_string_methods_for_params():
    from openvino.runtime.utils.cli_parser import path_to_str_or_object, str_list_to_str, \
        mean_scale_value_to_str, source_target_layout_to_str, layout_param_to_str, transform_param_to_str, \
        extensions_to_str_or_extensions_class, batch_to_int, transformations_config_to_str
    return {
        'input_model': path_to_str_or_object,
        'output': str_list_to_str,
        'mean_values': mean_scale_value_to_str,
        'scale_values': mean_scale_value_to_str,
        'source_layout': source_target_layout_to_str,
        'target_layout': source_target_layout_to_str,
        'layout': layout_param_to_str,
        'transform': transform_param_to_str,
        'extensions': extensions_to_str_or_extensions_class,
        'batch': batch_to_int,
        'transformations_config': transformations_config_to_str,
        'saved_model_tags': str_list_to_str
    }
