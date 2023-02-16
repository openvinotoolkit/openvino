# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

class Registry:
    """ The class is needed for supervising algorithms due to loading from the config file """

    def __init__(self, name):
        """ Creates Registry(...) instance for some specified algo family
        :param name: name of the algo family
         """
        self._name = name
        self._registry_dict = dict()

    def register(self, name=None):
        """ Puts algorithm to register collection to carry out order of calls and
        config validation used as decorator on algorithm classes.
        :param name: name of class to register
         """
        def _register(obj_name, obj):
            if obj_name in self._registry_dict:
                raise KeyError('{} is already registered in {}'.format(name, self._name))
            self._registry_dict[obj_name] = obj

        def wrap(obj):
            cls_name = name
            if cls_name is None:
                cls_name = obj.__name__
            _register(cls_name, obj)
            return obj

        return wrap

    def get(self, name):
        """ Get algorithm and metadata
        :param name: required algorithm name
        :return algo instance
         """
        if name not in self._registry_dict:
            raise KeyError('{} is unknown type of {} '.format(name, self._name))
        return self._registry_dict[name]

    @property
    def registry_dict(self):
        return self._registry_dict

    @property
    def name(self):
        return self._name


class RegistryStorage:
    def __init__(self, d):
        regs = [r for r in d.values() if isinstance(r, Registry)]
        self.registries = {}
        for r in regs:
            if r.name in self.registries:
                raise RuntimeError('There are more than one registry with the name "{}"'.format(r.name))
            self.registries[r.name] = r

    def get_registry(self, registry_name):
        if registry_name not in self.registries:
            raise RuntimeError('Cannot find registry with registry_name "{}"'.format(registry_name))
        return self.registries[registry_name]
