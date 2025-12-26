import json
import os

from conan.api.conan_api import ConanAPI
from conan.api.input import UserInput
from conan.api.model import MultiPackagesList, RecipeReference
from conan.api.output import cli_out_write, ConanOutput
from conan.api.subapi.audit import CONAN_CENTER_AUDIT_PROVIDER_NAME
from conan.cli import make_abs_path
from conan.cli.args import common_graph_args, validate_common_graph_args
from conan.cli.command import conan_command, conan_subcommand
from conan.cli.formatters.audit.vulnerabilities import text_vuln_formatter, json_vuln_formatter, \
    html_vuln_formatter
from conan.cli.printers import print_profiles
from conan.cli.printers.graph import print_graph_basic
from conan.errors import ConanException
from conan.internal.util.files import load


def _add_provider_arg(subparser):
    subparser.add_argument("-p", "--provider", help="Provider to use for scanning")


def _parse_error_threshold(result: dict, error_level: float) -> None:
    """Mark the result as error if any of the vulnerabilities has a severity greater than or equal to
       The error_level reflects the severity level configured by the user. As it uses float and CVSS
       score is limited to 10.0, users can use any higher number to skip it.

       :param result: Conan audit scan result. It's expected to find cvss there
       :param error_level: Threshold to raise an error in case of matching the severity level
       :return: None
    """
    if "conan_error" not in result:
        for ref in result["data"]:
            if "vulnerabilities" not in result["data"][ref]:
                continue
            for edge in result["data"][ref]["vulnerabilities"]["edges"]:
                preferred_base_score = float(edge["node"]["cvss"].get("preferredBaseScore", 0.0))
                if preferred_base_score >= error_level:
                    result.update(
                        {"conan_error":
                             f"The package {ref} has a CVSS score {preferred_base_score} and "
                             f"exceeded the threshold severity level {error_level}."})
                    break


@conan_subcommand(formatters={"text": text_vuln_formatter,
                              "json": json_vuln_formatter,
                              "html": html_vuln_formatter})
def audit_scan(conan_api: ConanAPI, parser, subparser, *args) -> dict:
    """
    Scan a given recipe for vulnerabilities in its dependencies.
    """
    common_graph_args(subparser)
    # Needed for the validation of args, but this should usually be left as False here
    # TODO: Do we then want to hide it in the --help?
    subparser.add_argument("--build-require", action='store_true', default=False,
                           help='Whether the provided reference is a build-require')
    subparser.add_argument("-sl", "--severity-level", action="store", default=9.0, type=float,
                           help="Set threshold for severity level to raise an error. "
                                "By default raises an error for any critical CVSS (9.0 or higher). "
                                " Use 100.0 to disable it.")
    subparser.add_argument("--context", help="Context to scan, by default both contexts are scanned "
                                             "if not specified",
                           choices=["host", "build"], default=None)

    _add_provider_arg(subparser)
    args = parser.parse_args(*args)

    # This comes from install command

    validate_common_graph_args(args)
    # basic paths
    cwd = os.getcwd()
    path = conan_api.local.get_conanfile_path(args.path, cwd, py=None) if args.path else None

    # Basic collaborators: remotes, lockfile, profiles
    remotes = conan_api.remotes.list(args.remote) if not args.no_remote else []
    overrides = eval(args.lockfile_overrides) if args.lockfile_overrides else None
    lockfile = conan_api.lockfile.get_lockfile(lockfile=args.lockfile, conanfile_path=path, cwd=cwd,
                                               partial=args.lockfile_partial, overrides=overrides)
    profile_host, profile_build = conan_api.profiles.get_profiles_from_args(args)
    print_profiles(profile_host, profile_build)

    # Graph computation (without installation of binaries)
    gapi = conan_api.graph
    if path:
        deps_graph = gapi.load_graph_consumer(path, args.name, args.version, args.user, args.channel,
                                              profile_host, profile_build, lockfile, remotes,
                                              # TO DISCUSS: defaulting to False the is_build_require
                                              args.update, is_build_require=args.build_require)
    else:
        deps_graph = gapi.load_graph_requires(args.requires, args.tool_requires, profile_host,
                                              profile_build, lockfile, remotes, args.update)
    print_graph_basic(deps_graph)
    deps_graph.report_graph_error()

    if deps_graph.error:
        return {"conan_error": deps_graph.error}

    provider = conan_api.audit.get_provider(args.provider or CONAN_CENTER_AUDIT_PROVIDER_NAME)

    scan_result = conan_api.audit.scan(deps_graph, provider, args.context)
    _parse_error_threshold(scan_result, args.severity_level)
    return scan_result


@conan_subcommand(formatters={"text": text_vuln_formatter,
                              "json": json_vuln_formatter,
                              "html": html_vuln_formatter})
def audit_list(conan_api: ConanAPI, parser, subparser, *args):
    """
    List the vulnerabilities of the given reference.
    """
    input_group = subparser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("reference", help="Reference to list vulnerabilities for", nargs="?")
    input_group.add_argument("-l", "--list", help="Package list file to list vulnerabilities for")
    input_group.add_argument("-s", "--sbom", help="SBOM file to list vulnerabilities for")
    input_group.add_argument("-lock", "--lockfile", help="Path to the lockfile to check for vulnerabilities")
    subparser.add_argument("-r", "--remote", help="Remote to use for listing")
    _add_provider_arg(subparser)
    args = parser.parse_args(*args)

    provider = conan_api.audit.get_provider(args.provider or CONAN_CENTER_AUDIT_PROVIDER_NAME)

    if args.list:
        listfile = make_abs_path(args.list)
        multi_package_list = MultiPackagesList.load(listfile)
        cache_name = "Local Cache" if not args.remote else args.remote
        package_list = multi_package_list[cache_name]
        refs_to_list = package_list.serialize()
        references = list(refs_to_list.keys())
        if not references:  # the package list might contain only refs, no revs
            ConanOutput().warning("Nothing to list, package list does not contain recipe revisions")
    elif args.sbom:
        sbom_file = make_abs_path(args.sbom)
        sbom = json.loads(load(sbom_file))
        if sbom.get("bomFormat") != "CycloneDX":
            raise ConanException(f"Unsupported SBOM format, only CycloneDX is supported.")
        purls = [component["purl"] for component in sbom["components"]]
        references = [purl.split("pkg:conan/")[1].replace("@", "/") for purl in purls]
    elif args.lockfile:
        lock_file = make_abs_path(args.lockfile)
        lockfile = conan_api.lockfile.get_lockfile(lock_file).serialize()
        references = [str(RecipeReference.loads(ref[0] if isinstance(ref, tuple) else ref))
                      for ref in lockfile["requires"]]
    else:
        references = [args.reference]
    return conan_api.audit.list(references, provider)


def _text_provider_formatter(providers_action):
    providers = providers_action[0]
    action = providers_action[1]

    if action == "remove":
        cli_out_write("Provider removed successfully.")
    elif action == "add":
        cli_out_write("Provider added successfully.")
    elif action == "auth":
        cli_out_write("Provider authentication added.")
    elif action == "list":
        if not providers:
            cli_out_write("No providers found.")
        else:
            for provider in providers:
                if provider:
                    cli_out_write(f"{provider.name} (type: {provider.type}) - {provider.url}")


def _json_provider_formatter(providers_action):
    ret = []
    for provider in providers_action[0]:
        if provider:
            ret.append({"name": provider.name, "url": provider.url, "type": provider.type})
    cli_out_write(json.dumps(ret, indent=4))


@conan_subcommand(formatters={"text": _text_provider_formatter, "json": _json_provider_formatter})
def audit_provider(conan_api, parser, subparser, *args):
    """
    Manage security providers for the 'conan audit' command.
    """

    subparser.add_argument("action", choices=["add", "list", "auth", "remove"],
                           help="Action to perform from 'add', 'list' , 'remove' or 'auth'")
    subparser.add_argument("name", help="Provider name", nargs="?")

    subparser.add_argument("--url", help="Provider URL")
    subparser.add_argument("--type", help="Provider type", choices=["conan-center-proxy", "private"])
    subparser.add_argument("--token", help="Provider token")
    args = parser.parse_args(*args)

    if args.action == "add":
        if not args.name or not args.url or not args.type:
            raise ConanException("Name, URL and type are required to add a provider")
        if " " in args.name:
            raise ConanException("Name cannot contain spaces")
        conan_api.audit.add_provider(args.name, args.url, args.type)

        if not args.token:
            user_input = UserInput(conan_api.config.get("core:non_interactive"))
            ConanOutput().write(f"Please enter a token for {args.name} the provider: ")
            token = user_input.get_password()
        else:
            token = args.token

        provider = conan_api.audit.get_provider(args.name)
        if token:
            conan_api.audit.auth_provider(provider, token)

        return [provider], args.action
    elif args.action == "remove":
        if not args.name:
            raise ConanException("Name required to remove a provider")
        conan_api.audit.remove_provider(args.name)
        return [], args.action
    elif args.action == "list":
        providers = conan_api.audit.list_providers()
        return providers, args.action
    elif args.action == "auth":
        if not args.name:
            raise ConanException("Name is required to authenticate on a provider")
        if not args.token:
            user_input = UserInput(conan_api.config.get("core:non_interactive"))
            ConanOutput().write(f"Please enter a token for {args.name} the provider: ")
            token = user_input.get_password()
        else:
            token = args.token

        provider = conan_api.audit.get_provider(args.name)
        conan_api.audit.auth_provider(provider, token)
        return [provider], args.action


@conan_command(group="Security")
def audit(conan_api, parser, *args):  # noqa
    """
    Find vulnerabilities in your dependencies.
    """
