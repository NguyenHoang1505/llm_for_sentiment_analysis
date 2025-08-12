import re
def remove_url(text):
    text = re.sub(r'(http\S+)?(\w+\.)+\S+', r'', text)
    return text
def convert_upper_to_lower(text):
    return text.lower()
def remove_special_character(text):
    text = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]', r' ', text)
    return text
def remove_duplicate_character(text):
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'(\s)\1+', r'\1', text)
    return text
def remove_acronyms(text, teencode):
    for x in teencode: 
        text = re.sub(fr' {x[0]} |^{x[0]} | {x[0]}$', fr' {x[1]} ', text)
    return text
def truncate(text):
    text = text.split()
    text = text[:900]
    return " ".join(text)