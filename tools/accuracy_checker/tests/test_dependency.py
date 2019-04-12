"""
Copyright (c) 2019 Intel Corporation

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

from accuracy_checker.dependency import ClassProvider, get_opts


def test_get_opts_positional_and_kwargs():
    opts = {'o': ((1,), {'a': 1})}
    args, kwargs = get_opts(opts['o'])

    assert args == (1,)
    assert kwargs == {'a': 1}


def test_get_opts_kwargs_only():
    opts = {'o': {'a': 1}}
    args, kwargs = get_opts(opts['o'])

    assert args == ()
    assert kwargs == {'a': 1}


def test_get_opts_positional_only():
    opts = {'o': (1, 2, 3)}
    args, kwargs = get_opts(opts['o'])

    assert args == (1, 2, 3)
    assert kwargs == {}


def test_class_provider():
    class BaseService(ClassProvider):
        __provider_type__ = 'Service'

    class ServiceA(BaseService):
        __provider__ = 'service_a'

    class ServiceB(BaseService):
        __provider__ = 'service_b'

    assert issubclass(ServiceA, BaseService)
    assert issubclass(ServiceB, BaseService)

    assert 'service_a' in BaseService.providers
    assert 'service_b' in BaseService.providers


def test_provide():
    class BaseService(ClassProvider):
        __provider_type__ = 'service'

        def __init__(self):
            pass

    class ServiceA(BaseService):
        __provider__ = 'service_a'

    provided = BaseService.provide('service_a')

    assert isinstance(provided, ServiceA)


def test_provide_with_args():
    class BaseService(ClassProvider):
        __provider_type__ = 'service'

        def __init__(self, bar):
            self.bar = bar

    class ServiceA(BaseService):
        __provider__ = 'service_a'

    provided = BaseService.provide('service_a', bar=42)

    assert isinstance(provided, ServiceA)
    assert provided.bar == 42
