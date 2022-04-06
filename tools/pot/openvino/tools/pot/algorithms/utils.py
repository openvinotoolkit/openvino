# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

def process_ignored_scope(ignored_scope):
    """ Returns ignored scope with names containing all nested subgraphs for nodes
    :param ignored_scope: ignored_scope from algorithm specific config
    :return ignored scope with fullnames for nodes
     """
    ignored_scope_with_fullnames = []
    for layer in ignored_scope:
        if isinstance(layer, (tuple, list)):
            ignored_scope_with_fullnames.append('|'.join(layer))
        else:
            ignored_scope_with_fullnames.append(layer)
    return ignored_scope_with_fullnames
