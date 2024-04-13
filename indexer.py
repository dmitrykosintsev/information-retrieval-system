import math
import os
import re
import sqlite3
import string
import sys
import time
from porterStemmer import PorterStemmer
from collections import Counter
from typing import TextIO

# the database is a simple dictionnary
database = {}

# NEW create an array with 73 positions for stop words
stopWords = []


# NEW The function loads stop words from a file into the array
def stop_list_gen():
    try:
        absolute_path = os.path.dirname(__file__)
        word_list = "stopwords.txt"
        full_path = os.path.join(absolute_path, word_list)
        file: TextIO = open(full_path, 'r')
    except IOError:
        print("Error in file")
        return False
    else:
        for l in file.readlines():
            stopWords.append(l)
        # for l in stopWords: #code to check the length of the array
        #     print(l)
        # print(len(stopWords))
    file.close()


# regular expression for: extract words, extract ID from path, check for hexa value
chars = re.compile(r'\W+')
pattid = re.compile(r'(\d{3})/(\d{3})/(\d{3})')

# the higher ID
tokens = 0
documents = 0
terms = 0

# NEW counters for ignored words
ignoredStopWords = 0
ignoredShortWords = 0
ignoredNumbers = 0
ignoredPunc = 0


#
# We will create a term object for each unique instance of a term
#

class Term():
    termid = 0
    tf_norm = 0  # Normalised term frequency
    idf = 0
    tfidf = {}
    docs = 0  # This is document frequency
    docids = {}  # Tuples of docid and term frequency per document


# split on any chars
def splitchars(line):
    return chars.split(line)


# process the tokens of the source code
def parse_token(line):
    global documents
    global tokens
    global terms
    global ignoredStopWords
    global ignoredShortWords
    global ignoredNumbers
    global ignoredPunc

    # this replaces any tab characters with a space character in the line
    # read from the file
    line = line.replace('\t', ' ')
    line = line.strip()

    #
    # This routine splits the contents of the line into tokens
    l = splitchars(line)

    # NEW Creating a PorterStemmer object to use in the for-loop
    ps = PorterStemmer()

    # for each token in the line process
    for elmt in l:

        # This statement removes the newline character if found
        elmt = elmt.replace('\n', '')

        # This statement converts all letters to lower case
        lower_elmt = elmt.lower().strip()

        #
        # Increment the counter of the number of tokens processed. This value will
        # provide the total size of the corpus in terms of the number of terms in the
        # entire collection
        #
        tokens += 1

        # NEW If the token satisfies at least one condition, we stop processing the current token

        # NEW check if the token is 2 or fewer characters long
        if len(lower_elmt) < 3:
            # print("Found a short word: ", lowerElmt)
            ignoredShortWords += 1
            continue

        # NEW check if the token in the list of stop words
        if lower_elmt in stopWords:
            # print("Found a stop word: ", lowerElmt)
            ignoredStopWords += 1
            continue

        # NEW check if the token is a number
        if lower_elmt.isdigit():
            # print("Found a number: ", lowerElmt)
            ignoredNumbers += 1
            continue

        # NEW Check if the token begins with punctuation
        if lower_elmt in string.punctuation:
            # print("Found a word starting with punctuation: ", lowerElmt)
            ignoredPunc += 1
            continue

        # Implementing stemming
        # print("Before stemming: ", lowerElmt)
        lower_elmt = ps.stem(lower_elmt, 0, len(lower_elmt) - 1)
        # print("After stemming: ", lowerElmt)

        # if the term doesn't currently exist in the term dictionary
        # then add the term
        if not (lower_elmt in database.keys()):
            terms += 1
            database[lower_elmt] = Term()
            database[lower_elmt].termid = terms
            database[lower_elmt].docids = dict()
            database[lower_elmt].tfidf = dict()
            database[lower_elmt].docs = 0

        # if the document is not currently in the postings
        # list for the term then add it
        #
        if not (documents in database[lower_elmt].docids.keys()):
            database[lower_elmt].docs += 1
            database[lower_elmt].docids[documents] = 0
            database[lower_elmt].tfidf[documents] = 0

        # Increment the counter that tracks the term frequency
        database[lower_elmt].docids[documents] += 1
        # database[lowerElmt].termfreq[documents] += 1

    return l


#
# Open and read the file line by line, parsing for tokens and processing. All of the tokenizing
# is done in the parsetoken() function. You should design your indexer program keeping the tokenizing
# as a separate function that will process a string as you will be able to reuse code for
# future assignments
#

def process(filename):
    try:
        file = open(filename, 'r')
    except IOError:
        print("Error in file %s" % filename)
        return False
    else:
        for l in file.readlines():
            parse_token(l)
    file.close()


#
# This function will scan through the specified directory structure selecting
# every file for processing by the tokenizer function
# Notices how this is a recursive function in that it calls itself to process
# sub-directories.
#

def walk_dir(cur, dirname):
    global documents
    all = {}
    all = [f for f in os.listdir(dirname) if
           os.path.isdir(os.path.join(dirname, f)) or os.path.isfile(os.path.join(dirname, f))]
    for f in all:
        if os.path.isdir(dirname + '/' + f):
            walkdir(cur, dirname + '/' + f)
        # ignore any *.dat file in the directory (back-comparable with U2 script)
        if f.endswith('.dat'):
            continue
        else:
            documents += 1
            cur.execute("insert into DocumentDictionary values (?, ?)", (dirname + '/' + f, documents))
            process(dirname + '/' + f)
    return True


# NEW CALCULATIONS OF TF-IDF BELOW

def compute_tf_idf(database):
    global documents
    N = documents

    # Compute the term frequency (IDF) for each word in each document
    for k, v in database.items():
        v.idf = math.log(N / v.docs)

        # Compute the TF-IDF score for each word
        for doc_id, term_freq in v.docids.items():
            if doc_id in v.tfidf:
                # NEW Normalise the term frequency
                v.tf_norm = (1 + math.log(term_freq, 10))
                v.tfidf[doc_id] = float(v.tf_norm * v.idf)

    # NEW Normalising tf-idf
    euc_norm = 0

    # NEW Euclidean norm
    for k, v in database.items():
        for d, f in v.tfidf.items():
            # print(f)
            euc_norm = euc_norm + math.pow(f, 2)
    # print("Before: ", Euc_norm) # Debuging purposes
    euc_norm = math.sqrt(euc_norm)
    # print("After: ", Euc_norm) # Debuging purposes

    # NEW Calculate the normalised vector
    for k, v in database.items():
        for d, f in v.tfidf.items():
            v.tfidf[d] = f / euc_norm
            # print(v.tfidf[d]) # Debuging purposes


"""
==========================================================================================
>>> main

This section is the 'main' or starting point of the indexer program.
The python interpreter will find this 'main' routine and execute it first.
==========================================================================================
"""

if __name__ == "__main__":
    #
    # Capture the start time of the routine so that we can determine the total running
    # time required to process the corpus
    #
    t0 = time.localtime()
    t0ns = time.time_ns()
    print('Processing Start Time: %.2d:%.2d:%.2d' % (t0.tm_hour, t0.tm_min, t0.tm_sec))

    # Load stop words into an array
    stop_list_gen()

    #
    # The corpus of documents must be extracted from the zip file and placed into the C:\corpus
    # directory or another directory that you choose. If you use another directory make sure that
    # you point folder to the appropriate directory.
    #
    folder_path = os.path.dirname(__file__)
    folder = os.path.join(folder_path, "cacm")

    #
    # Create a sqlite database to hold the inverted index. The isolation_level statment turns
    # on autocommit which means that changes made in the database are committed automatically
    #
    db_path = os.path.dirname(__file__)
    db_name = os.path.join(folder_path, "index.db")
    con = sqlite3.connect(db_name)
    con.isolation_level = None
    cur = con.cursor()

    #
    # In the following section three tables and their associated indexes will be created.
    # Before we create the table or index we will attempt to drop any existing tables in
    # case they exist
    #
    # Document Dictionary Table
    cur.execute("drop table if exists DocumentDictionary")
    cur.execute("drop index if exists idxDocumentDictionary")
    cur.execute("create table if not exists DocumentDictionary (DocumentName text, DocId int)")
    cur.execute("create index if not exists idxDocumentDictionary on DocumentDictionary (DocId)")

    # Term Dictionary Table
    cur.execute("drop table if exists TermDictionary")
    cur.execute("drop index if exists idxTermDictionary")
    cur.execute("create table if not exists TermDictionary (Term text, TermId int)")
    cur.execute("create index if not exists idxTermDictionary on TermDictionary (TermId)")

    # Postings Table
    cur.execute("drop table if exists Posting")
    cur.execute("drop index if exists idxPosting1")
    cur.execute("drop index if exists idxPosting2")
    cur.execute("create table if not exists Posting (TermId int, DocId int, tfidf real, docfreq int, termfreq int)")
    cur.execute("create index if not exists idxPosting1 on Posting (TermId)")
    cur.execute("create index if not exists idxPosting2 on Posting (Docid)")

    #
    # The walkdir method essentially executes the indexer. The walkdir method will
    # read the corpus directory, Scan all files, parse tokens, and create the inverted index.
    #
    walk_dir(cur, folder)
    compute_tf_idf(database)

    t1 = time.localtime()
    t1ns = time.time_ns()
    print('Indexing Complete, write to disk: %.2d:%.2d:%.2d' % (t1.tm_hour, t1.tm_min, t1.tm_sec))

    #
    # Create the inverted index tables.
    #
    # Insert a row into the TermDictionary for each unique term along with a termid which is
    # a integer assigned to each term by incrementing an integer
    # NEW
    for k, v in database.items():
        term_pair = [k, v.termid]
        cur.execute('INSERT INTO TermDictionary (Term, TermId) VALUES (?, ?)', term_pair)

    # Insert a row into the posting table for each unique combination of Docid and termid
    # NEW
    for k, v in database.items():
        term = v.termid
        docs = v.docs
        tf_wt = v.tf_norm
        for docid, termfreq in v.docids.items():
            row_values = (term, docid, v.tfidf[docid], tf_wt, docs)
            cur.execute('INSERT INTO Posting (TermId, DocId, tfidf, termfreq, docfreq) VALUES (?, ?, ?, ?, ?)',
                        row_values)
        # for docid, tfidf in v.tfidf.items():
        #     temp = tfidf
        #     cur.execute('INSERT INTO Posting (tfidf) VALUES (?)', [temp])

    #
    # Commit changes to the database and close the connection
    #
    con.commit()
    con.close()

    #
    # Print processing statistics
    #
    print("\nDocuments %i" % documents)
    print("Total number of tokens parsed is %i" % tokens)
    print("Total number of unique terms found and added to the index is %i" % terms)
    print("\nTokens that did not meet conditions and were not added to the dictionary:")
    print("Stop words %i" % ignoredStopWords)
    print("Short words %i" % ignoredShortWords)
    print("Numbers %i" % ignoredNumbers)
    print("Words starting with punctuation %i" % ignoredPunc)
    t2 = time.localtime()
    t2ns = time.time_ns()
    print('\nProcessing End Time: %.2d:%.2d:%.2d' % (t2.tm_hour, t2.tm_min, t2.tm_sec))
    print("Indexing time: ", end='')
    print("%d ms" % ((t1ns - t0ns) / 1000000))
    print("Overall running time: ", end='')
    print("%d ms" % ((t2ns - t0ns) / 1000000))

    # Code to print the database
    # print("\nPrinting the dictionary: ")
    # for k, v in database.items():
    #     print("Key: ", k, " | TermID: ", v.termid, "| Term Freq: ", v.tf_norm, "| iDF: ", v.idf, " | Docs: ", v.docs,
    #           " | DocIDs: ", v.docids, " | tf-idf: ", v.tfidf)
