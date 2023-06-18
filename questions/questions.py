import nltk
import sys
import os 
import numpy as np
import math

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    cwd = os.getcwd()
    folder = os.path.join(cwd, directory)
    map = dict()
    for file in os.listdir(folder):
        with open(os.path.join(folder, file)) as f:
            text = f.read()
            map[file] = text
    return map

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    
    #initial list of words
    wordList = word_tokenize(document)

    #convert to lowercase
    lowercaseList = []
    for word in wordList:
        lower = word.lower()
        lowercaseList.append(lower)
    
    #remove punctuation and stopwords
    punctuationList = list(punctuation)
    stopwordList = list(stopwords.words('english'))
    exclude = punctuationList + stopwordList
    final = []
    for word in lowercaseList:
        if word not in exclude:
            final.append(word)
    
    return final


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    
    docs = list(documents.keys())
    numDocs = len(docs)
    
    #counts how many times words appear across documents
    wordAppearanceCounter = dict()
    for doc1 in docs:
        for doc2 in docs:
            if doc1 != doc2:
                for word in documents[doc1]:
                    wordAppearanceCounter[word] = 1 #at minumum, word appears in doc1
                    if word in documents[doc2]:
                        wordAppearanceCounter[word] += 1

    #computes IDF
    IDF = dict()
    for key in wordAppearanceCounter:
        word = key
        numAppearances = wordAppearanceCounter[key]
        IDF[word] = np.log(numDocs/numAppearances)

    return IDF    

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """

    #compute sum of tf-idf values for each document
    tfidfSum = dict()
    for key in files:
        tfidfSum[key] = 0
        for word1 in query:
            #count times word in query appears in each document
            appearanceCounter = 0
            for word2 in files[key]:
                if word1 == word2:
                    appearanceCounter += 1
            #multiply by idf of the word and add to sum
            tfidfSum[key] += appearanceCounter * idfs[word1]
    
    #sort by relevance in reverse order according to td-idf sum
    sortedFiles = dict(sorted(tfidfSum.items(), key=lambda x: x[1], reverse=True))
    final = list(sortedFiles.keys())
    #get top n files
    return final[0:n]
    

        

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    
    #compute sum of idf values for each sentence
    idfSum = dict()
    for key in sentences:
        idfSum[key] = 0
        for word in query:
            if word in sentences[key]:
                idfSum[key] += idfs[word]
    
    #compute query term density for each sentence
    qtd = dict()
    for key in sentences:
        numWords = len(sentences[key])
        queryWordsMatches = 0
        for word in query:
            if word in sentences[key]:
                queryWordsMatches += 1
        qtd[key] = queryWordsMatches/numWords
    
    #create a sorted list of sentences by best choice (max idf, max qtd if tie)
    choices = []
    numSentences = len(list(sentences.keys()))
    
    while len(choices) < numSentences:
        #finds max idf sum of remaining choices
        maxidf = float('-inf')
        for key in idfSum:
            if idfSum[key] > maxidf and key not in choices:
                maxidf = idfSum[key]
        #finds best choice
        bestChoice = None
        for key1 in idfSum:
            if math.isclose(idfSum[key1], maxidf):
                #naive best choice
                bestChoice = key1
                #checks for ties
                for key2 in idfSum:
                    if key1 != key2 and math.isclose(idfSum[key1], idfSum[key2]):
                        maxqtd = max(qtd[key1], qtd[key2])
                        if qtd[key2] == maxqtd:
                            bestChoice = key2
        
        #adds best choice to list of sentences
        choices.append(bestChoice)
    
    #returns n top sentences
    return choices[0:n]




if __name__ == "__main__":
    main()
