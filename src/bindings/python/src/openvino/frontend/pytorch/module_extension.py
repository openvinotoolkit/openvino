# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

class ModuleExtension:
    def __init__(self, module, target_op, evaluate=None, convert=None, condition=None):
        """Create an extension that replaces a PyTorch module with a single op.

        This functionality works with PyTorch models only. A module can be
        identified by its type (e.g., `torch.nn.Linear`), module instance in
        the model, or module name.

        Args:
            module (str, torch.nn.Module, type(torch.nn.Module)): PyTorch
                module to replace.

            target_op (str): A target operation that will be used as a replacer
                for the module. It could be the name of the extension operation
                or an existing PyTorch operation (with `prim::` or `aten::`
                prefix following TorchScript syntax).

            evaluate (callable): A function with the signature
                `evaluate(module, *args, **kwargs)`. It replaces the target
                module in model execution and is responsible for producing
                valid output for the module to allow correct model tracing. By
                default, it calls the original module's forward method with
                the same arguments. The provided code will not be part of the
                final traced model; it is used only to produce valid results
                during tracing.

            convert (callable): A function with the signature
                `convert(target_op, *args, **kwargs)`. It is traced and becomes
                part of the final model instead of the target module. It
                accepts `target_op` as the first parameter, which appears as a
                single node in the graph, with the type of the node being the
                `target_op` provided as another argument above.

            condition (callable): A function with the signature
                `condition(module)`. It returns a boolean indicating whether
                the extension applies to the given module.
        """
        self.module = module
        self.target_op = target_op
        self.evaluate = evaluate
        if self.evaluate is None:
            self.evaluate = lambda module, *args, **kwargs: module(*args, **kwargs)
        self.convert = convert
        if self.convert is None:
            self.convert = lambda module, target_op, *args, **kwargs: target_op(*args, **kwargs)
        self.condition = condition
        if self.condition is None:
            self.condition = lambda module: True
