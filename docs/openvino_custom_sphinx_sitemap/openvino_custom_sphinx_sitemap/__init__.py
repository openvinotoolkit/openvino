import xml.etree.ElementTree as ET
from sphinx_sitemap import setup as base_setup, get_locales, hreflang_formatter


def setup(app):
    app.add_config_value(
        'ov_sitemap_urlset',
        default=None,
        rebuild=''
    )
    
    app.add_config_value(
        'ov_sitemap_meta',
        default=None,
        rebuild=''
    )

    setup = base_setup(app)
    for listener in app.events.listeners['build-finished']:
        if listener.handler.__name__ == 'create_sitemap':
            app.disconnect(listener.id)
    
    app.connect('build-finished', create_sitemap)
    return setup


def create_sitemap(app, exception):
    """Generates the sitemap.xml from the collected HTML page links"""

    urlset = app.builder.config.ov_sitemap_urlset
    meta = app.builder.config.ov_sitemap_meta

    site_url = app.builder.config.site_url or app.builder.config.html_baseurl
    site_url = site_url.rstrip('/') + '/'
    if not site_url:
        print("sphinx-sitemap error: neither html_baseurl nor site_url "
              "are set in conf.py. Sitemap not built.")
        return
    if (not app.sitemap_links):
        print("sphinx-sitemap warning: No pages generated for %s" %
              app.config.sitemap_filename)
        return

    ET.register_namespace('xhtml', "http://www.w3.org/1999/xhtml")

    root = ET.Element("urlset")

    if not urlset:
        root.set("xmlns", "http://www.sitemaps.org/schemas/sitemap/0.9")
    else:
        for item in urlset:
            root.set(*item)

    get_locales(app, exception)

    if app.builder.config.version:
        version = app.builder.config.version + '/'
    else:
        version = ""

    for link in app.sitemap_links:
        url = ET.SubElement(root, "url")
        scheme = app.config.sitemap_url_scheme
        if app.builder.config.language:
            lang = app.builder.config.language + '/'
        else:
            lang = ""

        ET.SubElement(url, "loc").text = site_url + scheme.format(
            lang=lang, version=version, link=link
        )

        if meta:
            for entry in meta:
                namespace, values = entry
                namespace_element = ET.SubElement(url, namespace)
                for tag_name, tag_value in values.items():
                    ET.SubElement(namespace_element, tag_name).text = tag_value

        if len(app.locales) > 0:
            for lang in app.locales:
                lang = lang + '/'
                linktag = ET.SubElement(
                    url,
                    "{http://www.w3.org/1999/xhtml}link"
                )
                linktag.set("rel", "alternate")
                linktag.set("hreflang",  hreflang_formatter(lang.rstrip('/')))
                linktag.set("href", site_url + scheme.format(
                    lang=lang, version=version, link=link
                ))

    filename = app.outdir + "/" + app.config.sitemap_filename
    ET.ElementTree(root).write(filename,
                               xml_declaration=True,
                               encoding='utf-8',
                               method="xml")
    print("%s was generated for URL %s in %s" % (app.config.sitemap_filename,
          site_url, filename))
