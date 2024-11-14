import xml.etree.ElementTree as ET
import queue
from pathlib import Path
from sphinx_sitemap import setup as base_setup, get_locales, hreflang_formatter, add_html_link, record_builder_type
from sphinx.util.logging import getLogger

logger = getLogger(__name__)

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
        
    app.connect("builder-inited", record_builder_type)
    app.connect("html-page-context", add_html_link)
    app.connect('build-finished', create_sitemap)
    app.parallel_safe = True
    app.parallel_read_safe = True
    app.parallel_write_safe = True
    return setup


def create_sitemap(app, exception):
    """Generates the sitemap.xml from the collected HTML page links"""

    urlset = app.builder.config.ov_sitemap_urlset
    meta = app.builder.config.ov_sitemap_meta

    site_url = app.builder.config.site_url

    if site_url:
        site_url.rstrip("/") + "/"
    else:
        logger.warning(
            "sphinx-sitemap: html_baseurl is required in conf.py." "Sitemap not built.",
            type="sitemap",
            subtype="configuration",
        )
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

    locales = get_locales(app)

    if app.builder.config.version:
        version = app.builder.config.version + '/'
    else:
        version = ""

    unique_links = set()
    while True:
        try:
            link = app.env.app.sitemap_links.get_nowait()  # type: ignore
            if link in unique_links:
                continue
            unique_links.add(link)
        except queue.Empty:
            break

        url = ET.SubElement(root, "url")
        
        if app.builder.config.language:
            lang = app.builder.config.language + "/"
        else:
            lang = ""

        scheme = app.config.sitemap_url_scheme 
        ET.SubElement(url, "loc").text = site_url + scheme.format(
            lang=lang, version=version, link=link
        )

        process_coveo_meta(meta, url, link)

        for lang in locales:
            lang = lang + "/"
            ET.SubElement(
                url,
                "{http://www.w3.org/1999/xhtml}link",
                rel="alternate",
                hreflang=hreflang_formatter(lang.rstrip("/")),
                href=site_url + scheme.format(lang=lang, version=version, link=link),
            )

    filename = Path(app.outdir) / app.config.sitemap_filename
    ET.ElementTree(root).write(filename,
                               xml_declaration=True,
                               encoding='utf-8',
                               method="xml")
    print("%s was generated for URL %s in %s" % (app.config.sitemap_filename,
          site_url, filename))

def process_coveo_meta(meta, url, link):
    if not meta:
        return

    for namespace, values in meta:
        namespace_element = ET.SubElement(url, namespace)

        for tag_name, tag_value in values.items():
            if tag_name == 'ovdoctype':
                processed_link = process_link(link)
                ET.SubElement(namespace_element, tag_name).text = processed_link
            else:
                ET.SubElement(namespace_element, tag_name).text = tag_value

def process_link(link):
    if '/' in link:
        return link.split('/')[0].replace("-", " ")
    return link.split('.html')[0].replace("-", " ")