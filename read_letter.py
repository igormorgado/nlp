from poppler import load_from_file, PageRenderer

document = load_from_file("carta.pdf")
text = []
for page_index in range(document.pages):
    page = document.create_page(page_index)
    page_text = page.text().replace('\f', '')
    text.append(page_text)

text = ' '.join(text)
text = text.split('\n')
text = '\n'.join(text[7:])

with open('carta.txt', 'w') as fd:
    fd.write(text)

fd.close()
    
