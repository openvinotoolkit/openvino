# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from e2e_tests.common.decorators import wrap_ord_dict


@wrap_ord_dict
def classification_comparators(device, postproc=None, target_layers=None, precision=None, a_eps=None, r_eps=None,
                               ntop=10):
    if postproc is None:
        postproc = {}
    return [("classification", {"device": device,
                                "ntop": ntop,
                                "precision": precision,
                                "a_eps": a_eps,
                                "r_eps": r_eps,
                                "postprocessors": postproc,
                                "target_layers": target_layers
                                }
             ),
            ("eltwise", {"device": device,
                         "a_eps": a_eps,
                         "r_eps": r_eps,
                         "precision": precision,
                         "target_layers": target_layers,
                         "ignore_results": True}
             )]


@wrap_ord_dict
def object_detection_comparators(device, postproc=None, precision=None, a_eps=None, r_eps=None, p_thr=0.5, iou_thr=None,
                                 mean_only_iou=False, target_layers=None):
    if postproc is None:
        postproc = {}
    return "object_detection", {"device": device,
                                "p_thr": p_thr,
                                "a_eps": a_eps,
                                "r_eps": r_eps,
                                "precision": precision,
                                "iou_thr": iou_thr,
                                "postprocessors": postproc,
                                "mean_only_iou": mean_only_iou,
                                "target_layers": target_layers
                                }


@wrap_ord_dict
def eltwise_comparators(device, postproc=None, precision=None, a_eps=None, r_eps=None,
                        target_layers=None, ignore_results=False, mean_r_eps=None):
    if postproc is None:
        postproc = {}
    return "eltwise", {"device": device,
                       "a_eps": a_eps,
                       "r_eps": r_eps,
                       "mean_r_eps": mean_r_eps,
                       "precision": precision,
                       "postprocessors": postproc,
                       "ignore_results": ignore_results,
                       "target_layers": target_layers
                       }




@wrap_ord_dict
def segmentation_comparators(device, postproc=None, precision=None, thr=None, target_layers=None):
    if postproc is None:
        postproc = {}
    return "semantic_segmentation", {"device": device,
                                     "thr": thr,
                                     "postprocessors": postproc,
                                     "target_layers": target_layers,
                                     "precision": precision
                                     }


@wrap_ord_dict
def dummy_comparators():
    return "dummy", {}


@wrap_ord_dict
def ssim_comparators(device, postproc=None, precision=None, thr=None, target_layers=None):
    if postproc is None:
        postproc = {}
    return "ssim", {"device": device,
                    "thr": thr,
                    "precision": precision,
                    "postprocessors": postproc,
                    "ignore_results": False,
                    "target_layers": target_layers
                    }


@wrap_ord_dict
def ssim_4d_comparators(device, postproc=None, precision=None, thr=None, target_layers=None, win_size=None):
    if postproc is None:
        postproc = {}
    return "ssim_4d", {"device": device,
                       "ssim_4d_thr": thr,
                       "precision": precision,
                       "postprocessors": postproc,
                       "ignore_results": False,
                       "target_layers": target_layers,
                       "win_size": win_size
                       }


@wrap_ord_dict
def ocr_comparators(device, postproc=None, precision=None, target_layers=None, top_paths=10, beam_width=10):
    if postproc is None:
        postproc = {}
    return "ocr", {"device": device,
                   "precision": precision,
                   "postprocessors": postproc,
                   "target_layers": target_layers,
                   "top_paths": top_paths,
                   "beam_width": beam_width
                   }
