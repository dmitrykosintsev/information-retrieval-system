"""
This file contains an example search engine that will search the inverted index
"""
import math
import re
import sqlite3
import time

# use simple dictionary data structures in Python to maintain lists with hash keys
docs = {}
results_list = {}
term = {}

# regular expression or: extract words, extract ID rom path, check or hexa value
chars = re.compile(r'\W+')
patt_id = re.compile(r'(\d{3})/(\d{3})/(\d{3})')


#
# Docs class: Used to store information about each unit document. In this is the Term object which stores each
# unique instance of termid or a docid.
#
class Docs:
    terms = {}


#
# Term class: used to store information or each unique termid
#
class Term:
    doc_freq = 0
    term_freq = 0
    idf = 0.0
    tfidf = 0.0


# split on any chars
def split_chars(line):
    return chars.split(line)


# this small routine is used to accumulate query idf values
def elen_q(elen, a):
    return (float(math.pow(a.idf, 2)) + float(elen))


# this small routine is used to accumulate document tfidf values
def elen_d(elen, a):
    return (float(math.pow(a.tfidf, 2)) + float(elen))


"""
================================================================================================
>>> main

This section is the 'main' or starting point o the indexer program. The python interpreter will find this 'main' routine and execute it first.
================================================================================================       """
if __name__ == '__main__':

    #
    # Create a sqlite database to hold the inverted index. The isolation_level statement turns
    # on autocommit which means that changes made in the database are committed automatically
    #
    con = sqlite3.connect("index.db")  # Path to be edited
    con.isolation_level = None
    cur = con.cursor()

    #
    #
    #
    line = input('Enter the search terms, each separated by a space: ')

    #
    # Capture the start time of the search so that we can determine the total running
    # time required to process the search
    #
    t0 = time.localtime()
    print('Processing Start Time: %.2d:%.2d:%.2d' % (t0.tm_hour, t0.tm_min, t0.tm_sec))

    #
    # This routine splits the contents of the line into tokens
    l = split_chars(line)

    #
    # Get the total number of documents in the collection
    #
    q = "select count(*) from DocumentDictionary"
    cur.execute(q)
    row = cur.fetchone()
    documents = row[0]

    # Initialize maxterms variable. This will be used to determine the maximum number of search
    # terms that exists in any one document.
    #
    maxterms = float(0)

    # process the tokens (search terms) entered by the user
    for elmt in l:
        # This statement removes the newline character if found
        elmt = elmt.replace('\n', '')

        # This statement converts all letters to lower case
        lower_elmt = elmt.lower().strip()

        #
        # Execute query to determine if the term exists in the dictionary
        #
        q = "select count(*) from TermDictionary where Term = '%s'" % (lower_elmt)
        cur.execute(q)
        row = cur.fetchone()

        #
        # If the term exists in the dictionary retrieve all documents for the term and store in a list
        #
        if row[0] > 0:
            q = "select distinct Docid, tfidf, docfreq, termfreq, posting.TermId FROM TermDictionary,Posting where Posting.TermId = TermDictionary.TermId and Term = '%s' order by Docid, Posting.TermId" % (
                lower_elmt)
            cur.execute(q)
            for row in cur.fetchall():

                i_termid = row[4]
                i_docid = row[0]

                if not (i_docid in docs.keys()):
                    docs[i_docid] = Docs()
                    docs[i_docid].terms = {}

                if not (i_termid in docs[i_docid].terms.keys()):
                    docs[i_docid].terms[i_termid] = Term()
                    docs[i_docid].terms[i_termid].doc_freq = row[2]
                    docs[i_docid].terms[i_termid].term_freq = row[3]
                    docs[i_docid].terms[i_termid].idf = 0.0
                    docs[i_docid].terms[i_termid].tfidf = 0.0

    #
    # Calculate tfidf values for both the query and each document

    # Each query:
    query_tfidf = {}
    for term in l:
        # Calculate tf value
        tf = 1 + math.log(l.count(term) / len(l), 10)

        # Calculate idf value

        q = "select count(*) from TermDictionary where Term = '%s'" % (lower_elmt)
        cur.execute(q)
        row = cur.fetchone()
        # print("Row is ", row)
        if row[0] > 0:
            idf = math.log(documents + 1 / (row[0] + 1)) + 1
        else:
            idf = 0.0
        # Calculate tf-idf value
        tfidf = tf * idf
        # Add tf-idf value to query vector
        query_tfidf[term] = tfidf

    # print("query_tfidf before normalization: ", query_tfidf)

    # Normalising tf-idf
    Euc_norm = 0

    for k, v in query_tfidf.items():
        Euc_norm = Euc_norm + math.pow(v, 2)

    Euc_norm = math.sqrt(Euc_norm)

    for k, v in query_tfidf.items():
        if v == 0:
            query_tfidf[k] = 0
        else:
            query_tfidf[k] = v / Euc_norm

    # print("query_tfidf after normalization: ", query_tfidf)
    # print("Docs is: ", docs[1].terms.keys().__contains__('html'))
    # print("Docs is: ", docs[1].terms.values())

    # Each document - calculated by the indexer
    document_tfidf = {}
    for doc_id in docs:
        document_tfidf[doc_id] = {}
        for term in l:
            if term in docs[doc_id].terms:
                print("Entering the loop")
                # Calculate tf value
                tf = docs[doc_id].terms[term].term_freq / docs[doc_id].terms[term].doc_freq
                # Calculate idf value
                idf = math.log(len(docs) / docs[doc_id].terms[term].docs)
                # Calculate tf-idf value
                tfidf = tf * idf
                # Add tf-idf value to document vector
                document_tfidf[doc_id][term] = tfidf
    # print("document_tfidf is: ", document_tfidf)

    # Using the tfidf (or weight) value, accumulate the vectors and calculate
    # the cosine similarity between the query and each document
    cosine_similarity = {}
    for doc_id in document_tfidf:
        # Calculate dot product of query and document vectors
        # print("query_tfidf[term] is: ", query_tfidf[term], "; term is: ", term)
        # print("document_tfidf[doc_id][term] is: ", document_tfidf[doc_id][term], "; term is: ", term)
        dot_product = sum(
            query_tfidf[term] * document_tfidf[doc_id][term] for term in query_tfidf if term in document_tfidf[doc_id])
        # Calculate magnitudes of query and document vectors
        query_magnitude = math.sqrt(sum(query_tfidf[term] ** 2 for term in query_tfidf))
        document_magnitude = math.sqrt(sum(document_tfidf[doc_id][term] ** 2 for term in document_tfidf[doc_id]))

        doc_terms = docs[doc_id].terms.keys()
        doc_length = 0.0

        for term_id in doc_terms:
            term_freq = docs[doc_id].terms[term_id].term_freq
            doc_freq = docs[doc_id].terms[term_id].doc_freq
            idf = docs[doc_id].terms[term_id].idf
            tfidf = term_freq * idf
            doc_length += tfidf ** 2

        doc_length = math.sqrt(doc_length)

        # Calculate cosine similarity
        try:
            cosine_similarity[doc_id] = dot_product / (query_magnitude * document_magnitude)
            # print("Trying")
        except:
            cosine_similarity[doc_id] = 0

    # print("cosine_similarity is: ", cosine_similarity)
    keylist = list(cosine_similarity.items())
    # print(keylist)

    # sort in descending order
    keylist.sort(reverse=True)
    i = 0

    # print out the top 20 most relevant documents (or as many as were found)
    print("Fetching results...")
    time.sleep(3)
    print("No.\t Document")
    print("---\t --------")
    for key, value in keylist:
        i += 1
        if i > 20:
            break
        q = "select DocumentName from documentdictionary where docid = '%d'" % (key)
        cur.execute(q)
        row = cur.fetchone()
        print(i, "\t", "%s has relevance of %f" % (row[0], float(key) / 10000))
    con.close()

    # Print ending time to show the processing duration of the query.
    t1 = time.localtime()
    t1ns = time.time_ns()
    print("\nDocuments searched: ", documents)
    print('Search complete: %.2d:%.2d:%.2d' % (t1.tm_hour, t1.tm_min, t1.tm_sec))
    print("Overall running time: ", end='')
    print("%d ms" % ((t1ns - t1ns) / 1000000))
