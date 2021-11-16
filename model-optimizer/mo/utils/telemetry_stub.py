# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


class Telemetry(object):
    """
    Stab file for the Telemetry class which is used when Telemetry class is not available.
    """

    def __init__(self, *arg, **kwargs):
        pass

    def send_event(self, *arg, **kwargs):
        print("XXXXXXX send_event")
        print(arg)
        print(kwargs)
        pass

    def send_error(self, *arg, **kwargs):
        print("XXXXXXX send_error")
        print(arg)
        print(kwargs)
        pass

    def start_session(self, *arg, **kwargs):
        print("XXXXXXX start_session")
        print(arg)
        print(kwargs)
        pass

    def end_session(self, *arg, **kwargs):
        print("XXXXXXX end_session")
        print(arg)
        print(kwargs)
        pass

    def force_shutdown(self, *arg, **kwargs):
        print("XXXXXXX force_shutdown")
        print(arg)
        print(kwargs)
        pass

    def send_stack_trace(self, *arg, **kwargs):
        print("XXXXXXX send_stack_trace")
        print(arg)
        print(kwargs)
        pass
