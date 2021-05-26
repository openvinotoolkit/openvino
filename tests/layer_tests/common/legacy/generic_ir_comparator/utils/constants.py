import os

try:
    mo_bin = os.environ['MO_ROOT']
    if not os.path.exists(mo_bin):
        raise EnvironmentError("Environment variable MO_ROOT points to non existing path {}".format(mo_bin))
except KeyError:
    pass
    # log.info("MO_ROOT variable is not set")

if os.environ.get('OUTPUT_DIR') is not None:
    out_path = os.environ['OUTPUT_DIR']
else:
    script_path = os.path.dirname(os.path.realpath(__file__))
    out_path = os.path.join(script_path, 'out')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

tf_models_path = os.path.join(out_path, 'tf_models')

