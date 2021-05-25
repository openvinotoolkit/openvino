import xml.etree.ElementTree as ET


class IR:
    def __init__(self, xml):
        if type(xml) is str:
            self.xml = ET.parse(xml)
        elif isinstance(xml, ET.ElementTree):
            self.xml = xml
        else:
            raise TypeError('Do not supported type for xml: {}'.format(type(xml)))

    def save(self, path):
        self.xml.write(path)
