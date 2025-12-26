from conan.internal.rest.client_routes import ClientV2Router
from conan.internal.util.files import sha1sum


def add_urls(package_list, remote):
    router = ClientV2Router(remote.url.rstrip("/"))
    for ref, packages in package_list.items():
        ref_info = package_list.recipe_dict(ref)
        for f, fp in ref_info.get("files", {}).items():
            ref_info.setdefault("upload-urls", {})[f] = {
                'url': router.recipe_file(ref, f), 'checksum': sha1sum(fp)
            }
        for pref in packages:
            pref_info = package_list.package_dict(pref)
            for f, fp in pref_info.get("files", {}).items():
                pref_info.setdefault("upload-urls", {})[f] = {
                    'url': router.package_file(pref, f), 'checksum': sha1sum(fp)
                }
