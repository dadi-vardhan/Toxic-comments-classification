import sys
sys.path.append('../')
from data_preprocessing import Data_Preprocessing

pp = Data_Preprocessing()

def test_remove_url():
    thestring   =   'The https://stackoverflow.com'
    p_text  =   pp.remove_urls(thestring)
    assert  p_text  ==  "The "

def test_remove_alpha_neumeric():
    thestring   =   '##%^#@!vis [hello]'
    p_text  =   pp.remove_alphanumeric(thestring)
    assert  p_text  ==  " vis hello "

def test_convert_lower_case():
    thestring   =   "HELLo worLD"
    p_text  =   pp.convert_lowercase(thestring)
    assert  p_text  ==  "hello world"

def test_split_sent():
    thestring  =    'HELLO worLD my name is ### vis99'
    p_words =   pp.split_comment(thestring)
    assert p_words  ==  ['HELLO',"worLD",'my','name','is','###','vis99']

def test_stem_and_lemmatize():
    thestring   =   'some people believes stemming is good.'
    p_text  =   pp.stemm_and_lemmatize(thestring)
    assert  p_text  == "peopl believ stem good."

def test_text_normalize():
    thestring   =   "Hello e-v-e-r-y-o-n-e!!!!@@@@!!!!! ?? @DONT BUY THIS , it to lab 6 month\
                    phone is dead dead, says!!! cheap components. I payed 400$ for LG G4.\
                    from https://www.amazon.com/gp/aw/ref=ya_awpi?UTF8=1, it's troubling!!"

    p_text  =   pp.text_normalize(thestring,tokens=False)
    assert  p_text  ==  "hello dont buy lab month phone dead dead say cheap compon pay 400 lg g4 troubl"



