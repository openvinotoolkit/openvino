import pkg_resources
import re
import sys
import tempfile
from itertools import islice

req_file=sys.argv[1]
con_file=sys.argv[2]

tmpfile = tempfile.TemporaryFile(mode="w+")

confile=open(con_file, mode='r')
for line in confile.readlines():
    tmpfile.write(line)
tmpfile.write("\n")
reqfile=open(req_file, mode='r')
for line in islice(reqfile, 2, None):
    tmpfile.write(line)

tmpfile.seek(0)

try:
    pkg_resources.require(tmpfile)
except Exception as inst:
    pattern_proto_version = re.compile(r"protobuf .*, Requirement.parse\('protobuf<=3\.20\.0,>=3\.1\.0'\), {'paddlepaddle'}")
    result_proto_version = pattern_proto_version.findall(str(inst))
    pattern_paddle_req= re.compile (r'paddlepaddle|numpy|six|protobuf')
    result_paddle_req = pattern_paddle_req.findall(str(inst))
    if len(result_proto_version) != 0:
        env = pkg_resources.Environment()
        env['protobuf'].clear()
        env.add(pkg_resources.DistInfoDistribution(project_name="protobuf", version="3.20.0"))
        ws = pkg_resources.working_set
        reqs = pkg_resources.parse_requirements(open(req_file, mode='r'))
        dists = ws.resolve(reqs, env, replace_conflicting=True)
    elif len(result_paddle_req) != 0:
        raise inst
finally:
    tmpfile.close()
    reqfile.close()
    confile.close()
