from pathlib import Path


def common_ir_generation(mo_runner, mo_out, model, precision, **kwargs):
    return ("get_ir", {"mo": {"mo_runner": mo_runner,
                              "mo_out": mo_out,
                              "model": model,
                              "precision": precision,
                              "additional_args": kwargs}})


def ir_pregenerated(xml, bin=None):
    if not bin:
        bin = str(Path(xml).with_suffix(".bin"))
    return "get_ir", {"pregenerated": {"xml": xml, "bin": bin}}


def ovc_ir_generation(mo_runner, mo_out, model, precision, **kwargs):
    return ("get_ir", {"get_ovc_model": {"mo_runner": mo_runner,
                                         "mo_out": mo_out,
                                         "model": model,
                                         "precision": precision,
                                         "additional_args": kwargs}})
