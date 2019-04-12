import benchmark

from argparse import ArgumentParser, SUPPRESS


def parse_args():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help=benchmark.HELP_MESSAGES["HELP"])
    args.add_argument('-i', '--path_to_images', type=str, required=True,
                      help=benchmark.HELP_MESSAGES['IMAGE_MESSAGE'])
    args.add_argument('-m', '--path_to_model', type=str, required=True,
                      help=benchmark.HELP_MESSAGES['MODEL_MESSAGE'])
    args.add_argument('-c', '--path_to_cldnn_config', type=str, required=False,
                      help=benchmark.HELP_MESSAGES['CUSTOM_GPU_LIBRARY_MESSAGE'])
    args.add_argument('-l', '--path_to_extension', type=str, required=False, default=None,
                      help=benchmark.HELP_MESSAGES['CUSTOM_GPU_LIBRARY_MESSAGE'])
    args.add_argument('-api', '--api_type', type=str, required=False, default='async', choices=['sync', 'async'],
                      help=benchmark.HELP_MESSAGES['API_MESSAGE'])
    args.add_argument('-d', '--target_device', type=str, required=False, default="CPU",
                      help=benchmark.HELP_MESSAGES['TARGET_DEVICE_MESSAGE'])
    args.add_argument('-niter', '--number_iterations', type=int, required=False, default=None,
                      help=benchmark.HELP_MESSAGES['ITERATIONS_COUNT_MESSAGE'])
    args.add_argument('-nireq', '--number_infer_requests', type=int, required=False, default=2,
                      help=benchmark.HELP_MESSAGES['INFER_REQUESTS_COUNT_MESSAGE'])
    args.add_argument('-nthreads', '--number_threads', type=int, required=False, default=None,
                      help=benchmark.HELP_MESSAGES['INFER_NUM_THREADS_MESSAGE'])
    args.add_argument('-b', '--batch_size', type=int, required=False, default=None,
                      help=benchmark.HELP_MESSAGES['BATCH_SIZE_MESSAGE'])
    args.add_argument('-pin', '--infer_threads_pinning', type=str, required=False, default='YES',
                      choices=['YES', 'NO'], help=benchmark.HELP_MESSAGES['INFER_THREADS_PINNING_MESSAGE'])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    benchmark.main(args)
