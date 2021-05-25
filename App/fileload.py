import docx
def readfromtxt(file):
    try:
        content = file.read().decode('utf-8')
        file.close()
        return content
    except IOError:
        error = 'There was an error opening the file!'
        return error

def readfromdocx(file):
    try:
        doc = docx.Document(file)
        content = '\n'.join(para.text for para in doc.paragraphs)
        return content

    except IOError:
        error = 'There was an error opening the file!'
        return error

def fileload(file):
    ext = file.filename.split('.')[1]
    if ext == 'txt':
        content = readfromtxt(file)
    elif ext == 'docx':
        content = readfromdocx(file)
    else:
        content = "Wrong file type"
    return content 
