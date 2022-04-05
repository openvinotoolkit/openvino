# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#! [mri_demo:demo]
import numpy as np
import cv2 as cv
import argparse
import time
from openvino.inference_engine import IECore


def kspace_to_image(kspace):
    assert(len(kspace.shape) == 3 and kspace.shape[-1] == 2)
    fft = cv.idft(kspace, flags=cv.DFT_SCALE)
    img = cv.magnitude(fft[:,:,0], fft[:,:,1])
    return cv.normalize(img, dst=None, alpha=255, beta=0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MRI reconstrution demo for network from https://github.com/rmsouza01/Hybrid-CS-Model-MRI (https://arxiv.org/abs/1810.12473)')
    parser.add_argument('-i', '--input', dest='input', help='Path to input .npy file with MRI scan data.')
    parser.add_argument('-p', '--pattern', dest='pattern', help='Path to sampling mask in .npy format.')
    parser.add_argument('-m', '--model', dest='model', help='Path to .xml file of OpenVINO IR.')
    parser.add_argument('-l', '--cpu_extension', dest='cpu_extension', help='Path to extensions library with FFT implementation.')
    parser.add_argument('-d', '--device', dest='device', default='CPU',
                        help='Optional. Specify the target device to infer on; CPU, '
                             'GPU, HDDL or MYRIAD is acceptable. For non-CPU targets, '
                             'HETERO plugin is used with CPU fallbacks to FFT implementation. '
                             'Default value is CPU')
    args = parser.parse_args()

    xml_path = args.model
    assert(xml_path.endswith('.xml'))
    bin_path = xml_path[:xml_path.rfind('.xml')] + '.bin'

    ie = IECore()
    ie.add_extension(args.cpu_extension, "CPU")

    net = ie.read_network(xml_path, bin_path)

    device = 'CPU' if args.device == 'CPU' else ('HETERO:' + args.device + ',CPU')
    exec_net = ie.load_network(net, device)

    # Hybrid-CS-Model-MRI/Data/stats_fs_unet_norm_20.npy
    stats = np.array([2.20295299e-01, 1.11048916e+03, 4.16997984e+00, 4.71741395e+00], dtype=np.float32)
    # Hybrid-CS-Model-MRI/Data/sampling_mask_20perc.npy
    var_sampling_mask = np.load(args.pattern)  # TODO: can we generate it in runtime?
    print('Sampling ratio:', 1.0 - var_sampling_mask.sum() / var_sampling_mask.size)

    data = np.load(args.input)
    num_slices, height, width = data.shape[0], data.shape[1], data.shape[2]
    pred = np.zeros((num_slices, height, width), dtype=np.uint8)
    data /= np.sqrt(height * width)

    print('Compute...')
    start = time.time()
    for slice_id, kspace in enumerate(data):
        kspace = kspace.copy()

        # Apply sampling
        kspace[var_sampling_mask] = 0
        kspace = (kspace - stats[0]) / stats[1]

        # Forward through network
        input = np.expand_dims(kspace.transpose(2, 0, 1), axis=0)
        outputs = exec_net.infer(inputs={'input_1': input})
        output = next(iter(outputs.values()))
        output = output.reshape(height, width)

        # Save predictions
        pred[slice_id] = cv.normalize(output, dst=None, alpha=255, beta=0, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

    print('Elapsed time: %.1f seconds' % (time.time() - start))

    WIN_NAME = 'MRI reconstruction with OpenVINO'

    slice_id = 0
    def callback(pos):
        global slice_id
        slice_id = pos

        kspace = data[slice_id]
        img = kspace_to_image(kspace)

        kspace[var_sampling_mask] = 0
        masked = kspace_to_image(kspace)

        rec = pred[slice_id]

        # Add a header
        border_size = 20
        render = cv.hconcat((img, masked, rec))
        render = cv.copyMakeBorder(render, border_size, 0, 0, 0, cv.BORDER_CONSTANT, value=255)
        cv.putText(render, 'Original', (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, color=0)
        cv.putText(render, 'Sampled (PSNR %.1f)' % cv.PSNR(img, masked), (width, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, color=0)
        cv.putText(render, 'Reconstructed (PSNR %.1f)' % cv.PSNR(img, rec), (width*2, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, color=0)

        cv.imshow(WIN_NAME, render)
        cv.waitKey(1)

    cv.namedWindow(WIN_NAME, cv.WINDOW_NORMAL)
    print(num_slices)
    cv.createTrackbar('Slice', WIN_NAME, num_slices // 2, num_slices - 1, callback)
    callback(num_slices // 2)  # Trigger initial visualization
    cv.waitKey()
#! [mri_demo:demo]
