""" Importing all the required Libraries
    If You any library missing goto cmd and type 
    "pip install <library name>"
"""

import csv
import pandas as pd
import PyPDF2
import os, os.path
from os import chdir, getcwd, listdir, path
import glob
from time import strftime
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import codecs
import math
from wordcloud import WordCloud
import matplotlib.pyplot as plt



""" This Function is Removing The Stop Words from the files """

def Remove_StopWords(data):
    print ("Removing the stop words from the files")
    data = data.split(" ")
    Stop_words =['abstract', 'introduction', 'conclusions', 'related work', 'author', 'university','a','able',
    'about','across','after','all','almost','also','am','among','an','and','any','are','as','at','be','because'
    ,'been','but','by','can','cannot','could','dear','did','do','does','either','else','ever','every','for','from'
    ,'get','got','had','has','have','he','her','hers','him','his','how','however','i','if','in','into','is','it',
    'its','just','least','let','like','likely','may','me','might','most','must','my','neither','no'
    ,'nor','not','of','off','often','on','only','or','other','our','own','rather','said',
    'say','says','she','should','since','so','some','than','that','the','their','them','then',
    'there','these','they','this','tis','to','too','twas','us','wants','was','we','were','what',
    'when','where','which','while','who','whom','why','will','with','would','yet','you','your','1','2','3','4','n','/','\n','\rn']
    another_list = []
    for x in data:
        if x in Stop_words:           
            """Removing the stop words from the file if any occur"""
            data.remove(x)  
    return data

""" This Function convert the pdf files into text files so that the manipulation is easy """
def PdftoText():
    # Getting all the files from the folder Documents
    list = os.listdir("Documents/")
    size=len(list)
    Paths=[]
    i=0
    # To complete the absolute path attaching directory to the file
    while (i<size):
        t=os.path.join("Documents/",list[i])
        Paths.append(t)
        i=i+1

    # Here making txt file extension with the same file name as pdf
    for item in Paths:
        path=item
        head,tail=os.path.split(path)

        var="\\"
        tail=tail.replace(".pdf",".txt")

        name=head+var+tail
        content = ""
        pdfFileObj = open(path, 'rb')
    
        # creating a pdf reader object
        pdf = PyPDF2.PdfFileReader(path,strict=False)
        for i in range(0, pdf.getNumPages()):
            #getting the content of pdf page i.e text
            content += pdf.getPage(i).extractText() + "\n"
        print (strftime("   %H:%M:%S"), " pdf  -> txt ")

        # Writing all the pdf data to txt files 
        with open(name,'ab') as out:
            out.write(content.encode("UTF-8"))
            out.close 

# Return the frequency of a word occured in document
def freq(word, doc):
    return doc.count(word)

# Return the total number of words in document
def word_count(doc):
    return len(doc)

# Return the Term frequency of word in document
def tf(word, doc):
    return (freq(word, doc) / float(word_count(doc)))

# Return the count of word in all documents
def num_docs_containing(word, list_of_docs):
    count = 0
    for document in list_of_docs:
        if freq(word, document) > 0:
            count += 1
    return 1 + count

# Return the inverse document frequency of word in all documents
def idf(word, list_of_docs):
    return math.log(len(list_of_docs) /
            float(num_docs_containing(word, list_of_docs)))

# Return the Term frequency *inverse document frequency of word in document into word in all documents
def tf_idf(word, doc, list_of_docs):
    return (tf(word, doc) * idf(word, list_of_docs))

# Return the 50 max Term frequency words in the documents 
def get_Fifty_max(lis):
    Lis=sorted(lis,key=lambda x:(-x[1],x[0]),reverse=True)    
    MaxList=[]
    for i in range(0,50):
        MaxList.append(Lis[i])
    return MaxList

# Return the 50 max Inverse document frequency words in the documents 
def idfMax(list_set,list_of_docs):
    size_set=len(list_set)
    List_tf=[]
    for r in range(0,size_set):
        maxl=[]
        for h in range(0,len(list_of_docs)):
            maxl.append(tf_idf(list_set[r],list_of_docs[h],list_of_docs))
        List_tf.append((list_set[r],max(maxl)))
    max_values=get_Fifty_max(List_tf)
    return max_values

# Return the 50 max Term frequency words in the documents 
def Max(list_set,list_of_docs):
    size_set=len(list_set)
    List_tf=[]
    
    for r in range(0,size_set):
        maxl=[]
        for h in range(0,len(list_of_docs)):
            maxl.append(tf(list_set[r],list_of_docs[h]))
        List_tf.append((list_set[r],max(maxl)))
    max_values=get_Fifty_max(List_tf)
    return max_values

# show the wordcloud for the most popular and non-popular words 
def Word_Cloud(text):
    wordcloud = WordCloud(max_font_size=40).generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def TextData():
    files = os.listdir("Documents/")
    size=len(files) 
    list_set=[]   
    t=0
    documents=[]
    list_of_docs=[]
    while (t<size):   
        if files[t].endswith(".txt"):
            ftxt=os.path.join("Documents/",files[t])     
            tfile=open(ftxt,"rb")
            line=tfile.read() # read the content of file and store in "line"
            
            line=str(line).lower()
            line=line.strip()
            table = str.maketrans({key: None for key in string.punctuation})
            line = line.translate(table)
            
            documents.extend(line)
            list_of_docs.append(line)
            tfile.close() # close the file
            line = Remove_StopWords(line)
            
            for l in line:
                list_set.append(l) # all words from all documents
        t+=1
    list_set  = [x for x in list_set if not x.startswith('x')] #remove the garbage data starting with 'x' in the list
    list_set=list(set(list_set))

    tfmax_values=Max(list_set,list_of_docs)

    tfidf_max=idfMax(list_set,list_of_docs)

    tfmax_values = sorted(tfmax_values, key=lambda tup: tup[1])  # sort the values in descending order for tf
    tfidf_max = sorted(tfidf_max, key=lambda tup: tup[1])        # sort the values in descending order for tfidf
    
    df=pd.DataFrame(tfmax_values)
    df.to_csv('tf_list.csv',index=False)                        #save the 50 words of tf in csv file
    values=[idx for idx, val in tfmax_values]
    
    df=pd.DataFrame(tfidf_max)
    df.to_csv('tfidf_list.csv',index=False)                      #save the 50 words of tfidf in csv file
    vals=[idx for idx, val in tfidf_max]
    
    return values,vals
       

#### Calling the Functions        
#NoTE: Once the files are converted you can comment down the pdftoText() function call 
PdftoText()                 # This function call convert the pdf files into txt

values,vals=TextData()      #Getting max 50 max values for both tf tfidf
Word_Cloud(str(values))     #generating word cloud for tf
Word_Cloud(str(vals))       #generating word cloud for tfidf
