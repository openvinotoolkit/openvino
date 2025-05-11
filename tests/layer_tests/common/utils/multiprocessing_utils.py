# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import os
import platform
import signal
import sys
import traceback
from multiprocessing import Process, Queue, TimeoutError, ProcessError
from queue import Empty as QueueEmpty
from typing import Callable, Union

if platform.system() == "Darwin":
    # Fix for MacOS
    import multiprocessing
    multiprocessing.set_start_method("forkserver", True)


def _mp_wrapped_func(func: Callable, func_args: list, queue: Queue, logger_queue: Queue):
    """
    Wraps callable object with exception handling. Current wrapper is a target for
    `multiprocessing_run` function
    :param func: see `multiprocessing_run`
    :param func_args: see `multiprocessing_run`
    :param queue: multiprocessing.Queue(). Used for getting callable object return values
    :param logger_queue: multiprocessing.Queue(). Used for getting logs from child process in parent process
    :return:
    """

    error_message = ""
    res = None
    try:
        res = func(*func_args)
    except:
        ex_type, ex_value, tb = sys.exc_info()
        error_message = "{tb}\n{ex_type}: {ex_value}".format(tb=''.join(traceback.format_tb(tb)),
                                                             ex_type=ex_type.__name__, ex_value=ex_value)
    queue.put((error_message, res))


def multiprocessing_run(func: Callable, func_args: list, func_log_name: str, timeout: Union[int, None] = None):
    """
    Wraps callable object to a separate process using multiprocessing module
    :param func: callable object
    :param func_args: list of arguments for callable
    :param func_log_name: name of callable used for logging
    :param timeout: positive int to limit execution time
    :return: return value (or values) from callable object
    """
    queue = Queue()
    logger_queue = Queue(-1)
    process = Process(target=_mp_wrapped_func, args=(func, func_args, queue, logger_queue))
    process.start()
    try:
        error_message, *ret_args = queue.get(timeout=timeout)
    except QueueEmpty:
        raise TimeoutError("{func} running timed out!".format(func=func_log_name))
    finally:
        queue.close()

        # Extract logs from Queue and pass to root logger
        while not logger_queue.empty():
            rec = logger_queue.get()
            log.getLogger().handle(rec)
        logger_queue.close()

        if process.is_alive():
            process.terminate()
            process.join()
        else:
            exit_signal = multiprocessing_exitcode_to_signal(process.exitcode)
            if exit_signal:
                raise ProcessError(
                    "{func} was killed with a signal {signal}".format(func=func_log_name, signal=exit_signal))

    if error_message:
        raise ProcessError("\n{func} running failed: \n{msg}".format(func=func_log_name, msg=error_message))

    ret_args = ret_args[0] if len(ret_args) == 1 else ret_args  # unwrap from list if only 1 item is returned
    return ret_args


def multiprocessing_exitcode_to_signal(exitcode):
    """
    Map multiprocessing exitcode to signals from "signal" module
    :param exitcode: multiprocessing exitcode
    :return: signal from "signal" if exitcode mapped on signal or None
    """
    # Multiprocessing return negative values of signal of the process, but on Win they are positive.
    # Bring the value to the positive format.
    exit_code = exitcode if os.name == "nt" else -exitcode
    if exit_code > 0:
        code_map = {int(getattr(signal, sig)): str(getattr(signal, sig))
                    for sig in dir(signal) if sig.startswith("SIG")}
        exit_signal = code_map[exit_code] if exit_code in code_map else exit_code
    else:
        exit_signal = None
    return exit_signal
