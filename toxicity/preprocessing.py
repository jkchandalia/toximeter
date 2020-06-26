

def remove_punctuation(text, exclude=["'"]):
    #Remove punctuation but leave apostrophe
    #TO DO: remove numbers
    if exclude:
        punctuation_to_remove = ''.join(list(set(string.punctuation)-set(exclude)))
    else:
        punctuation_to_remove = string.punctuation
    text = text.translate(str.maketrans('', '', punctuation_to_remove)
    return text

def remove_numbers(text):
    #Remove punctuation but leave apostrophe
    #TO DO: remove numbers
    text = text.translate(str.maketrans('', '', string.digits)
    return text

def keep_alpha_char(text):
    pass

def remove_stop_words(text):
    return ' '.join([word.strip() for word in text.split() if word not in stop])

def tokenize(text):
    return text.lower().split()