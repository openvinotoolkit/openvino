import re
filePath = "../../samples/sample3/main.cpp"
with open (filePath, 'r+' ) as f:
    content = f.read()
    a = '\\1[PREPROCESSED]\\2'
    contentNew = re.sub('(.*std::cout << \")(.*)', r'{}'.format(a), content, flags = re.M)
    f.seek(0)
    f.write(contentNew)
    f.close()
