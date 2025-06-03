import xml.etree.ElementTree as ET
import queue
from pathlib import Path
from sphinx_sitemap import setup as base_setup, add_html_link, record_builder_type
from sphinx.util.logging import getLogger

logger = getLogger(__name__)

def setup(app):
    app.add_config_value('ov_sitemap_urlset', default=None, rebuild='')
    app.add_config_value('ov_sitemap_meta', default=None, rebuild='')
    app.connect("builder-inited", record_builder_type)
    app.connect("html-page-context", add_html_link)
    app.connect('build-finished', lambda app, exc: create_sitemap(app, exc, ['google', 'coveo']))

    setup = base_setup(app)

    for listener in app.events.listeners['build-finished']:
        if listener.handler.__name__ == 'create_sitemap':
            app.disconnect(listener.id)
    
    app.parallel_safe = True
    app.parallel_read_safe = True
    app.parallel_write_safe = True
    return setup

def create_sitemap(app, exception, searchers):
    meta = app.builder.config.ov_sitemap_meta
    site_url = app.builder.config.site_url

    if site_url:
        site_url.rstrip("/") + "/"
    else:
        logger.warning("sphinx-sitemap: html_baseurl is required in conf.py. Sitemap not built.", type="sitemap", subtype="configuration")
        return

    if not app.sitemap_links:
        print(f"sphinx-sitemap warning: No pages generated.")
        return

    all_links = []
    while True:
        try:
            all_links.append(app.sitemap_links.get_nowait())
        except queue.Empty:
            break

    unique_links = set(all_links)

    for searcher in searchers:
        ET.register_namespace('xhtml', "http://www.w3.org/1999/xhtml")
        namespaces = {"xmlns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        if searcher == "coveo":
            namespaces["xmlns:coveo"] = "https://www.coveo.com/en/company/about-us"

        root = ET.Element("urlset", namespaces)
        version = app.builder.config.version + '/' if app.builder.config.version else ""

        for link in unique_links:
            url = ET.SubElement(root, "url")
            lang = app.builder.config.language + "/" if app.builder.config.language else ""
            scheme = app.config.sitemap_url_scheme
            ET.SubElement(url, "loc").text = site_url + scheme.format(lang=lang, version=version, link=link)

            if searcher == "coveo":
                process_coveo_meta(meta, url, link)
            elif searcher == "google":
                from datetime import datetime
                today_date = datetime.now().strftime('%Y-%m-%d')
                ET.SubElement(url, "lastmod").text = today_date
                ET.SubElement(url, "changefreq").text = "monthly"
                ET.SubElement(url, "priority").text = "0.5"

        filename = Path(app.outdir) / f"sitemap_{searcher}.xml"
        ET.ElementTree(root).write(filename, xml_declaration=True, encoding='utf-8', method="xml")
        print(f"sitemap_{searcher}.xml was generated for URL {site_url} in {filename}")


def process_coveo_meta(meta, url, link):
    if not meta:
        return
    
    for namespace, values in meta:
        namespace_element = ET.SubElement(url, namespace)
        loc_element = url.find("loc")
        
        for tag_name, tag_value in values.items():
            if tag_name == 'ovdoctype':
                ET.SubElement(namespace_element, tag_name).text = process_link(link)
            elif tag_name == 'ovcategory' and loc_element is not None:
                ET.SubElement(namespace_element, tag_name).text = extract_categories(loc_element.text)
            elif tag_name == 'ovversion':
                ET.SubElement(namespace_element, tag_name).text = tag_value

def process_link(link):
    if '/' in link:
        return format_segment(link.split('/')[0].replace("-", " "))
    return format_segment(link.split('.html')[0].replace("-", " "))

def extract_categories(link):
    path = link.split("://")[-1]
    segments = path.split('/')[1:]
    if segments and segments[-1].endswith('.html'):
        segments = segments[:-1]
    if segments:
        segments = segments[1:]
    if segments and '.' in segments[0]:
        year, *rest = segments[0].split('.')
        if year.isdigit() and len(year) == 4:
            segments[0] = year
    segments = [format_segment(segment) for segment in segments]
    if segments:
        hierarchy = ['|'.join(segments[:i]) for i in range(1, len(segments) + 1)]
        return ';'.join(hierarchy)
    return "No category"

def format_segment(segment):
    if segment == 'c_cpp_api': segment = 'C/C++_api'
    return ' '.join(word.capitalize() for word in segment.replace('-', ' ').replace('_', ' ').split())