import pkg_resources
import re
import sys

req_file=sys.argv[1]

try:
    pkg_resources.require(open(req_file, mode='r'))
except Exception as inst:
    pattern = re.compile(r"protobuf .*, Requirement.parse\('protobuf<=3\.20\.0,>=3\.1\.0'\), {'paddlepaddle'}")
    result = pattern.findall(str(inst))
    if len(result) == 0:
        raise inst
    else:
        env = pkg_resources.Environment()
        env['protobuf'].clear()
        env.add(pkg_resources.DistInfoDistribution(project_name="protobuf", version="3.20.0"))
        ws = pkg_resources.working_set
        reqs = pkg_resources.parse_requirements(open(req_file, mode='r'))
        dists = ws.resolve(reqs, env, replace_conflicting=True)
