# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import time
import torch
import logging as log


class BenchHook:
    def __init__(self):
        """Initialize the time list."""
        self.tm_list = []

    def clear_time_list(self):
        """Clear the time list."""
        self.tm_list.clear()

    def get_time_list(self):
        """Return the time list."""
        return self.tm_list

    def new_forward(self, model, model_type):
        """Define a new forward function."""
        org_forward = model.forward
        if model_type in ['decoder', 'codegen2', 'mpt', 'replit', 'chatglm', 'falcon']:
            def my_forward(input_ids: torch.LongTensor, attention_mask=None, past_key_values=None, **kwargs):
                beg = time.time()
                ret = org_forward(input_ids, attention_mask, past_key_values, **kwargs)
                end = time.time()
                self.tm_list.append(end - beg)
                return ret
            model.forward = my_forward
        elif model_type in ['t5', 'blenderbot', 'codet5']:
            def my_forward(input_ids=None, attention_mask=None, decoder_input_ids=None, encoder_outputs=None, past_key_values=None, **kwargs):
                beg = time.time()
                ret = org_forward(input_ids, attention_mask, decoder_input_ids, encoder_outputs, past_key_values, **kwargs)
                end = time.time()
                self.tm_list.append(end - beg)
                return ret
            model.forward = my_forward
        else:
            log.warning(f'model_type:{model_type}, does not support overloaded model forward.')
