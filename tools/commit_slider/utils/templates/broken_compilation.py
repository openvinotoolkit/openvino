from utils.templates.common_template import Template

class BrokenCompilationTemplate(Template):
    def __init__(self, cfg):
        # super().__init__(cfg)
        # self.createCash()
        pass

    def generateBrokenCompTemplate(cfg, customCfg):
        tmpJSON = cfg
        if '-v' in customCfg and customCfg['-v'] == 'false':
            tmpJSON['verboseOutput'] = False
        if '-c' in customCfg:
            tmpJSON['commitList']['getCommitListCmd'] = "git log {} --boundary --pretty=\"%h\"".format(customCfg['-c'])

        # if "errorPattern" in tmpl and\
        #     "precommitPath" in tmpl and\
        #     "testCmd" in tmpl:
        #     tmpJSON["runConfig"]["stopPattern"] = tmpl["errorPattern"]
        #     tmpJSON["dlbConfig"]["commonPath"] = tmpl["precommitPath"]
        #     tmpJSON["cachedPathConfig"]["commonPath"] = tmpl["precommitPath"]
        #     tmpJSON["appCmd"] = CfgManager.singlestepStrFormat(
        #         tmpJSON["appCmd"],
        #         tmpl["appCmd"],
        #         "testCmd"
        #     )
        # else:
        #     raise("Template is incomplete.")
        # subPath = "private_linux_manylinux2014_release/"
        # if "subPath" in tmpl:
        #     subPath = tmpl["subPath"]
        # tmpJSON["dlbConfig"]["subPath"] = subPath
        # tmpJSON["cachedPathConfig"]["subPath"] = subPath

        return tmpJSON