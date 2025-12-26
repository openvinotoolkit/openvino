import json
import datetime

from conan.api.conan_api import ConanAPI
from conan.api.model import ListPattern, MultiPackagesList
from conan.api.output import Color, cli_out_write
from conan.cli import make_abs_path
from conan.cli.command import conan_command, OnceArgument
from conan.cli.formatters.list import list_packages_html
from conan.errors import ConanException

# Keep them so we don't break other commands that import them, but TODO: Remove later
remote_color = Color.BRIGHT_BLUE
recipe_name_color = Color.GREEN
recipe_color = Color.BRIGHT_WHITE
reference_color = Color.WHITE
error_color = Color.BRIGHT_RED
field_color = Color.BRIGHT_YELLOW
value_color = Color.CYAN


def _format_timestamp_human(timestamp):
    # used by ref.repr_humantime() to print human readable time
    return datetime.datetime.fromtimestamp(int(timestamp),
                                           datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')


def print_serial(item, indent=None, color_index=None):
    indent = "" if indent is None else (indent + "  ")
    color_index = 0 if color_index is None else (color_index + 1)
    color_array = [Color.BRIGHT_BLUE, Color.BRIGHT_GREEN, Color.BRIGHT_WHITE,
                   Color.BRIGHT_YELLOW, Color.BRIGHT_CYAN, Color.BRIGHT_MAGENTA, Color.WHITE]
    color = color_array[color_index % len(color_array)]
    if isinstance(item, dict):
        for k, v in item.items():
            if isinstance(v, (str, int)):
                if k.lower() == "error":
                    color = Color.BRIGHT_RED
                    k = "ERROR"
                elif k.lower() == "warning":
                    color = Color.BRIGHT_YELLOW
                    k = "WARN"
                color = Color.BRIGHT_RED if k == "expected" else color
                color = Color.BRIGHT_GREEN if k == "existing" else color
                cli_out_write(f"{indent}{k}: {v}", fg=color)
            else:
                cli_out_write(f"{indent}{k}", fg=color)
                print_serial(v, indent, color_index)
    elif isinstance(item, type([])):
        for elem in item:
            cli_out_write(f"{indent}{elem}", fg=color)
    elif isinstance(item, int):  # Can print 0
        cli_out_write(f"{indent}{item}", fg=color)
    elif item:
        cli_out_write(f"{indent}{item}", fg=color)


def print_list_text(results):
    """ Do a little format modification to serialized
    package list, so it looks prettier on text output
    """
    info = results["results"]

    # Extract command single package name
    new_info = {}
    for remote, remote_info in info.items():
        new_remote_info = {}
        for ref, content in remote_info.items():
            if ref == "error":
                new_remote_info[ref] = content
            else:
                name, _ = ref.split("/", 1)
                new_remote_info.setdefault(name, {})[ref] = content
        new_info[remote] = new_remote_info
    info = new_info

    # TODO: The errors are not being displayed
    info = {remote: {"warning": "There are no matching recipe references"} if not values else values
            for remote, values in info.items()}

    def format_timestamps(item):
        if isinstance(item, dict):
            result = {}
            for k, v in item.items():
                if isinstance(v, dict) and v.get("timestamp"):
                    timestamp = v.pop("timestamp")
                    k = f"{k} ({_format_timestamp_human(timestamp)})"
                result[k] = format_timestamps(v)
            return result
        return item
    info = {remote: format_timestamps(values) for remote, values in info.items()}
    print_serial(info)


def print_list_compact(results):
    info = results["results"]

    """ transform the dictionary into a more compact one, keeping the internals
    but forming full recipe and package references including revisions at the top levels
    """
    for remote, remote_info in info.items():
        if not remote_info or "error" in remote_info:
            info[remote] = {"warning": "There are no matching recipe references"}
            continue
        prepare_pkglist_compact(remote_info)

    print_serial(info)


def prepare_pkglist_compact(pkglist):
    for ref, ref_info in pkglist.items():
        new_ref_info = {}
        for rrev, rrev_info in ref_info.get("revisions", {}).items():
            new_rrev = f"{ref}#{rrev}"
            timestamp = rrev_info.pop("timestamp", None)
            if timestamp:
                new_rrev += f"%{timestamp} ({_format_timestamp_human(timestamp)})"

            packages = rrev_info.pop("packages", None)
            if packages:
                for pid, pid_info in packages.items():
                    new_pid = f"{ref}#{rrev}:{pid}"
                    rrev_info[new_pid] = pid_info
            new_ref_info[new_rrev] = rrev_info
        pkglist[ref] = new_ref_info

    def compute_common_options(pkgs):
        """ compute the common subset of existing options with same values of a set of packages
        """
        result = {}
        all_package_options = [p.get("info", {}).get("options") for p in pkgs.values()]
        all_package_options = [p for p in all_package_options if p]  # filter pkgs without options
        for package_options in all_package_options:  # Accumulate all options for all binaries
            result.update(package_options)
        for package_options in all_package_options:  # Filter those not common to all
            result = {k: v for k, v in result.items()
                      if k in package_options and v == package_options[k]}
        for package_options in all_package_options:
            for k, v in package_options.items():
                if v != result.get(k):
                    result.pop(k, None)
        return result

    def compact_format_info(local_info, common_options=None):
        """ return a dictionary with settings and options in short form for compact format
        """
        result = {}
        settings = local_info.pop("settings", None)
        if settings:  # A bit of pretty order, first OS-ARCH
            values = [settings.pop(s, None)
                      for s in ("os", "arch", "build_type", "compiler")]
            values = [v for v in values if v is not None]
            values.extend(settings.values())
            result["settings"] = ", ".join(values)
        options = local_info.pop("options", None)
        if options:
            if common_options is not None:
                options = {k: v for k, v in options.items() if k not in common_options}
            options = ", ".join(f"{k}={v}" for k, v in options.items())
            options_tag = "options(diff)" if common_options is not None else "options"
            result[options_tag] = options
        for k, v in local_info.items():
            if isinstance(v, dict):
                v = ", ".join(f"{kv}={vv}" for kv, vv in v.items())
            elif isinstance(v, type([])):
                v = ", ".join(v)
            if v:
                result[k] = v
        return result

    def compact_diff(diffinfo):
        """ return a compact and red/green diff for binary differences
        """
        result = {}
        for k, v in diffinfo.items():
            if not v:
                continue
            if isinstance(v, dict):
                result[k] = {"expected": ", ".join(value for value in v["expected"]),
                             "existing": ", ".join(value for value in v["existing"])}
            else:
                result[k] = v
        return result

    for ref, revisions in pkglist.items():
        if not isinstance(revisions, dict):
            continue
        for rrev, prefs in revisions.items():
            pkg_common_options = compute_common_options(prefs)
            pkg_common_options = pkg_common_options if len(pkg_common_options) > 4 else None
            for pref, pref_contents in prefs.items():
                pref_info = pref_contents.pop("info", None)
                if pref_info is not None:
                    pref_contents.update(compact_format_info(pref_info, pkg_common_options))
                diff_info = pref_contents.pop("diff", None)
                if diff_info is not None:
                    pref_contents["diff"] = compact_diff(diff_info)


def print_list_json(data):
    results = data["results"]
    myjson = json.dumps(results, indent=4)
    cli_out_write(myjson)


@conan_command(group="Consumer", formatters={"text": print_list_text,
                                             "json": print_list_json,
                                             "html": list_packages_html,
                                             "compact": print_list_compact})
def list(conan_api: ConanAPI, parser, *args):
    """
    List existing recipes, revisions, or packages in the cache (by default) or the remotes.
    """
    parser.add_argument('pattern', nargs="?",
                        help="A pattern in the form 'pkg/version#revision:package_id#revision', "
                             "e.g: \"zlib/1.2.13:*\" means all binaries for zlib/1.2.13. "
                             "If revision is not specified, it is assumed latest one.")
    parser.add_argument('-p', '--package-query', default=None, action=OnceArgument,
                        help="List only the packages matching a specific query, e.g, os=Windows AND "
                             "(arch=x86 OR compiler=gcc)")
    parser.add_argument('-fp', '--filter-profile', action="append",
                        help="Profiles to filter the binaries")
    parser.add_argument('-fs', '--filter-settings', action="append",
                        help="Settings to filter the binaries")
    parser.add_argument('-fo', '--filter-options', action="append",
                        help="Options to filter the binaries")
    parser.add_argument("-r", "--remote", default=None, action="append",
                        help="Remote names. Accepts wildcards ('*' means all the remotes available)")
    parser.add_argument("-c", "--cache", action='store_true', help="Search in the local cache")
    parser.add_argument("-g", "--graph", help="Graph json file")
    parser.add_argument("-gb", "--graph-binaries", help="Which binaries are listed", action="append")
    parser.add_argument("-gr", "--graph-recipes", help="Which recipes are listed", action="append")
    parser.add_argument("-gc", "--graph-context", help="Filter the results by the given context",
                        default=None, action=OnceArgument,
                        choices=["build", "host", "build-only", "host-only"])
    parser.add_argument('--lru', default=None, action=OnceArgument,
                        help="List recipes and binaries that have not been recently used. Use a"
                             " time limit like --lru=5d (days) or --lru=4w (weeks),"
                             " h (hours), m(minutes)")

    args = parser.parse_args(*args)

    if args.pattern is None and args.graph is None:
        args.pattern = "*"  # All packages
    if args.graph:  # a few arguments are not compatible with this
        if args.pattern:
            raise ConanException("Cannot define both the pattern and the graph json file")
        if args.lru:
            raise ConanException("Cannot define lru when loading a graph json file")
        if args.filter_profile or args.filter_settings or args.filter_options:
            raise ConanException("Filtering binaries cannot be done when loading a graph json file")
    if (args.graph_recipes or args.graph_binaries) and not args.graph:
        raise ConanException("--graph-recipes and --graph-binaries require a --graph input")
    if args.remote and args.lru:
        raise ConanException("'--lru' cannot be used in remotes, only in cache")

    if args.graph:
        graphfile = make_abs_path(args.graph)
        pkglist = MultiPackagesList.load_graph(graphfile, args.graph_recipes, args.graph_binaries,
                                               context=args.graph_context)
    else:
        ref_pattern = ListPattern(args.pattern, rrev=None, prev=None)
        if not ref_pattern.package_id and (args.package_query or args.filter_profile or
                                           args.filter_settings or args.filter_options):
            raise ConanException("--package-query and --filter-xxx can only be done for binaries, "
                                 "a 'pkgname/version:*' pattern is necessary")
        if args.lru:
            if not ref_pattern.rrev and not ref_pattern.package_id:  # If package_id => #latest
                raise ConanException("'--lru' must be used with recipe revision pattern, "
                                     "use 'pkg/version#*' argument")
            if ref_pattern.package_id and not ref_pattern.prev:
                raise ConanException("'--lru' must be used with package revision pattern, "
                                     "use 'pkg/version:*#*' argument")
        # If neither remote nor cache are defined, show results only from cache
        pkglist = MultiPackagesList()
        profile = conan_api.profiles.get_profile(args.filter_profile or [],
                                                 args.filter_settings,
                                                 args.filter_options) \
            if args.filter_profile or args.filter_settings or args.filter_options else None
        if args.cache or not args.remote:
            try:
                cache_list = conan_api.list.select(ref_pattern, args.package_query, remote=None,
                                                   lru=args.lru, profile=profile)
            except Exception as e:
                pkglist.add_error("Local Cache", str(e))
            else:
                pkglist.add("Local Cache", cache_list)
        if args.remote:
            remotes = conan_api.remotes.list(args.remote)
            for remote in remotes:
                try:
                    remote_list = conan_api.list.select(ref_pattern, args.package_query, remote,
                                                        profile=profile)
                except Exception as e:
                    pkglist.add_error(remote.name, str(e))
                else:
                    pkglist.add(remote.name, remote_list)

    return {
        "results": pkglist.serialize(),
        "conan_api": conan_api,
        "cli_args": " ".join([f"{arg}={getattr(args, arg)}" for arg in vars(args) if getattr(args, arg)])
    }
