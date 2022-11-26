import re
filePath = "../../samples/sample3/main.cpp"
# filePath = 'test.txt'
with open (filePath, 'r+' ) as f:
    content = f.read()
    # print(content)
    # contentNew = re.sub('(\d{2})\/(\d{2})\/(\d{4})', r'\1-\2-\3', content, flags = re.M)
    a = '\\1[PREPROCESSED]\\2'
    contentNew = re.sub('(.*std::cout << \")(.*)', r'{}'.format(a), content, flags = re.M)
    f.seek(0)
    f.write(contentNew)
    f.close()
# import fileinput
# for line in fileinput.input("test.txt", inplace=True):
#     # replace every line according to regex
#     print('{} {}'.format(fileinput.filelineno(), line), end='') # for Python 3