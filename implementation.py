from nltk.stem.porter import PorterStemmer  # This is the porter steammer
import nltk  # This is the base package for the usage of the natural language processing
from nltk.tokenize import sent_tokenize, word_tokenize  # This is the sentence tokenizer
import os  # os module will be used to pick up all the files and the see through all those
import xml.etree.ElementTree as et  # will be used to parse the xml files .
from nltk.corpus import stopwords  # for stopwords removal
# import scipy.stats.binom
from scipy.stats.distributions import binom
import math  # for calculating the log function
import time  # Just for debugging .


# binom.cdf(successes, attempts, chance_of_success_per_attempt)
# For steaming the ibtauned tokens and then returning the list of the tokenz from the words
# AUTHOR : SHROUQ
# PROGRAMMER :SHROUQ & MUHAMMAD SHAFAY AMJAD
# Copyrights to : Shrouq & Shafay
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


# The below function is responsible for tokenizing as well as returning the right output.
def tokenize(text):
    tokens = nltk.word_tokenize(text)  # This tokenizes in to words
    stop_words = set(stopwords.words('english'))  # Gets the stop words in english
    tokens = [w for w in tokens if w.strip() not in stop_words]  # This removes the stop words
    stems = stem_tokens(tokens, PorterStemmer())  # This sent the list of words to the stemmer
    return stems


# for just having the tokenized sentences.
def tokenize_sentences(text):
    tokens = sent_tokenize(text)
    sentences = []
    for sent in tokens:
        sentences.append(tokenize(sent))
    return sentences


# This function will be responsible for the reading of the multiple files
# Parsing a single file and then returning the result string .
def parse_xmlfile_data(file_location):
    tree = et.parse(file_location)  # passing in the xml file location to be parsed.
    root = tree.getroot()  #
    # print(root.tag)  #Debugging
    # for child in root:
    #     print(child.tag,child.attrib)  # Printing the child tag and the child attributes .. .
    # print(root.child("Text").text)
    # document_text =  root.find('TEXT')
    # print(document_text.TEXT)
    # for do_text in root.findall("TEXT"):
    #    print(do_text.text)
    # do_text = root.find("TEXT")
    return root.find("TEXT").text  # This returns the text extracted from the TEXT , and th


# Getting all the files from the directory
def get_all_xml_files():
    xml_files = []  # A list which will be holding the destination of the xml files
    dir_path = os.path.dirname(
        os.path.realpath(__file__))  # Getting the Current directory of the script , where the file is placed
    # print(dir_path + "\\DUC2001") Debugging
    for filename in os.listdir(dir_path + "\\DUC2001"):
        if filename.startswith("AP"):  # just for Ap started files , can be done for all
            xml_files.append(
                dir_path + "\\DUC2001\\" + filename)  # This adds in the file names in the list of the xml files (All the documents)
    return xml_files  # returning the xml files to the main function


# Below function is responsible for getting the word counts of the specific document from the steamed words list
def count_words_in_document(document_steamed_words_list):
    word_count = {}  # dictionary to hold the steamed word counts
    for word in document_steamed_words_list:
        word = word.strip()  # this strips the extra spaces
        if word in word_count.keys():
            word_count[word] += 1
        else:  #
            word_count[word] = 1  # Intializing the word count to be equal to one,
    Total_words_count = sum(word_count.values())  # This adds in all the values and them sums them up
    # print(Total_words_count)
    return word_count, Total_words_count  # This returns the word count dictionary


# Calculate the probability of a word  in the document , (using the word_count_dictionary)
# pass dictionary of words of document counts  , total_words in the documents
def count_words_prob_in_doc(word_count_dictionary, total_word_count):
    document_word_probabilities = {}

    for word, count in word_count_dictionary.items():

        document_word_probabilities[word] = float(
            count) / total_word_count  # Adds in the probability of the words in the document.

    return document_word_probabilities  # This returns the words with the probabilities


# returns the binomial using the specified probabilities .
def binomial(n, c, p):
    return binom.cdf(c, n, p)


# n: size of the input
# c : count of the input
# p : probability of the input
def likelihood(n, c, p):
    return binomial(n, c, p)


# Hypothesis 1
# probability of word in document  = equals to probability of word in background corpus (then RETurn FALSE)
# Will recieve dictionaries in both parameters
# c1 : count of word in input
# c2: count of word in background corpus
# N1: size of input
# N2 : size of background corpus
# p1 : current_document dictionary holds the probabilities
# p2 : back ground corpus dictonary holds the probabilities.
def H1(c_1, c_2, n_1, n_2):
    p = float(c_1 + c_2) / float(n_1 + n_2)
    # print(p,p1,p2)
    # l_c1 = likelihood(n_1, c_1, p)
    # l_c2 = likelihood(n_2, c_2, p)
    # print(l_c1, l_c2)
    return likelihood(n_1, c_1, p) * likelihood(n_2, c_2, p)


# Hypothesis 2
# p1 : count of word in current document / total number of words .
# p2 : count of word in background corpus / total number of words in the background corpus.
def H2(c_1, c_2, n_1, n_2):
    p1 = float(c_1) / float(n_1)
    p2 = float(c_2) / float(n_2)
    return likelihood(n_1, c_1, p1) * likelihood(n_2, c_2, p2)


# To write in the file
def write_score_to_filedoc(score_list):
    output = open("output.txt","a+")

    for score in score_list:
        output.writelines(str(score) + "\n")
    print("score have been writen in the text file 'output.txt'")




# To test a single document with the probabilities.
def test_single():
    documents_loc_list = get_all_xml_files()  # documents locations list
    documents_data_list = list()  # all documents data lists
    steamed_documents_words_lists = list()  # steaemd document words lists [list of list]
    all_document_steamed_words = list()  # This contains all the documents steamed list
    documents_word_count_list = list()  # documents to hold word counts dictionaries in list
    document_data = parse_xmlfile_data(documents_loc_list[0])  # one document data
    documents_data_list.append(document_data)
    tk_doc_data = tokenize(document_data)
    for item in tk_doc_data:
        all_document_steamed_words.append(item)  # This contains all the steamed data from all the documents
    steamed_documents_words_lists.append(tk_doc_data)
    # print("The Total number of words in the file are : ",total_word_count)
    # print("Total number of unique words in the file are : ",len(word_count.items()))
    # print(words_counts.items())
    # print(steamed_documents_words_lists)
    words_counts, total_word_count = count_words_in_document(tk_doc_data)
    document_words_prob = count_words_prob_in_doc(words_counts, total_word_count)
    print(document_words_prob)
    # for i,j in words_counts.items():
    #     print(i,j)


# -------------------------------------------------------------------------------------------------------------

# This is the main function --  --
def main():
    documents_loc_list = get_all_xml_files()  # documents locations list

    steamed_documents_words_lists = list()  # steaemd document words lists [list of list]

    all_document_steamed_words = list()  # This contains all the documents steamed list

    documents_doc_sentences_list = []  # will hold list of document sentences .

    # Generating Background corpus , with counts and all others .
    docs_doc_sentence_string = dict()
    doc_sentences_string = dict()  # Will hold the string of the sentences .
    d_counter = 0
    for document_loc in documents_loc_list:
        document_data = parse_xmlfile_data(document_loc).lower()  # parse xml, get text, convert to lower.
        doc_sentences_string[d_counter] = [i.strip() for i in sent_tokenize(document_data)]
        docs_doc_sentence_string[d_counter] = doc_sentences_string # Adds in the sentence dictionary
        documents_doc_sentences_list.append(tokenize_sentences(parse_xmlfile_data(document_loc)))
        d_counter += 1
        tk_doc_data = tokenize(document_data)


        for item in tk_doc_data:
            all_document_steamed_words.append(item)

        steamed_documents_words_lists.append(tk_doc_data)

    total_background_words_counts, total_background_counts = count_words_in_document(
        all_document_steamed_words)

    counter = 1

    doc_dictionary = {}
    '''# BELOW HAS THE PROBABILITIES OF ALL THE DOCUMENTS WORDS'''
    # all_documents_words_prob = count_words_prob_in_doc(words_counts,total_word_count) # Sends in the total words count prob
    '''THE BELOW IS THE VERIFICATION OF THE PROBABILITIES OF WORDS COMBINED comes up to be one :D '''
    # print("The Total probability of all words in all documents is: ",sum(all_documents_words_prob.values()))
    # print("The amount of words involved are :",len(all_documents_words_prob.values()))
    # print("The Total documents are :",len(documents_prob_word_list))
    # print("Total number of words are",len(all_document_steamed_words))
    # print("Words in document 1 : ", len(documents_prob_word_list[0]))
    # print("some WORDS from all words in all documents :",list(all_documents_words_prob.keys())[0:5])
    # print("some probabilities from all words in all documents :",list(all_documents_words_prob.values())[0:5])
    # print(documents_prob_word_list[0].values())
    # all_doc_word_scores = []
    # print(documents_doc_sentences_list[0])

    # Steamed documnet words lists = [doc1wordlist, doc2word2list, doc3wordlist,.......]
    for document_word_list in steamed_documents_words_lists:
        inner_sentence_counter = 0

        doc_words_counts, doc_total_word_count = count_words_in_document(document_word_list)

        doc_word_lembda_score = {}

        for word in doc_words_counts.keys():

            lembda = (H1(doc_words_counts[word], total_background_words_counts[word], doc_total_word_count,
                         total_background_counts)) / (
                         H2(doc_words_counts[word], total_background_words_counts[word], doc_total_word_count,
                            total_background_counts))

            word_loglikelihood = -2 * math.log(lembda)

            if word_loglikelihood > 10.83:
                doc_word_lembda_score[word] = 1
                # print("Found one greater :: ", word, " ::than 10.83 in document : ", counter)
            else:
                doc_word_lembda_score[word] = 0

        doc_dictionary[
            counter - 1] = doc_word_lembda_score  # This keeps the record in the form of dictonaries. of the sentences.
        counter += 1
    print("Document processed : ",len(doc_dictionary))  # Total documents dictionary
    #print(len(doc_dictionary[0]))  # First Document dictionary
    #print(len(doc_dictionary[1]))

    doc_number = 0
    doc_sentence_score_list = dict() # Dictionary
    for sentence_list in documents_doc_sentences_list: #
        # print(doc_dictionary[doc_number]) # 1st document dictionary
        sentence_score = dict()
        sentence_number = 0
        for sentence in sentence_list:  # Sentences in document
            relative_sentence_score = 0.0
            abs_sentence_score = 0.0
            # print(sentence) # per sentence  ON WHICH SENTENCE LIST
            sentence_words_length = len(sentence)  # holds the length of the sentences
            for word in sentence:
                if word in doc_dictionary[doc_number].keys():
                    # print(doc_dictionary[doc_number][word]) print the values of the word
                    abs_sentence_score += doc_dictionary[doc_number][word]  # This adds in the values of the specified sentence

            relative_sentence_score = float(abs_sentence_score) / float(sentence_words_length) # This gets the sentence score

            sentence_score[sentence_number] = relative_sentence_score # This adds in the sentence score
            sentence_number += 1
        doc_sentence_score_list[doc_number] = sentence_score
        #print(doc_sentence_score_list)
        #if doc_number == 16:
            #break
        #break

        doc_number += 1
    print("sentence score Processing completed.")
    # print("Total number of documents :",len(documents_loc_list))
    # print("Total number of steamed words in all document :", len(all_document_steamed_words))
    # print("Total number of steamed words in document 1:",len(steamed_documents_words_lists[0]))  # prints the length of the first word list.

    # print(len(all_document_steamed_words))  # This contains all the Words in it .
    # print(len(documents_loc_list))  # Just for the debugging
    # print(len(documents_data_list))

    # string = "This is my first program and this is done using some of the specified worked place"
    # tokenized_steams  = tokenize(string)   # This tokenize the specific file in the form of the string
    # print(tokenized_steams)  # This prints the curent Tokenized stems in the form of the file .
    # Printing out the final vectors and the loops .
    docs_vector_list = list()
    for _doc, _value in doc_sentence_score_list.items():
        vector_list = list()
        print("#"*30, "IN DOCUMENT : ", _doc, "#"*30)
        for sentence, score in _value.items():
            if score == 0.00 and sum(_value.values()) == 0:
                vector_list.append(0.0)
            else:
                vector_list.append(score/float(sum(_value.values())))
            if score >0:
                print("Sentence no: ", sentence, " score :",score)
        docs_vector_list.append(vector_list)
# --------------------------------------------------------------------------------------------------------------------
    print("Vector list has been generated!")
    print(docs_vector_list[14])
    #print(doc_sentences_string[0])
    #ERROR CORRECTING BELOW
    print(doc_sentences_string[14][0])
    doc_count = 0
    for doc in docs_vector_list:
        print("===============================Doc : ", doc_count, " ========================================")
        sentence = 0
        #print(doc)
        for sent in doc:
            if sent > 0.0:
                print(docs_doc_sentence_string[doc_count][sentence] , sent)  # Prints the document sentence and the score (if topic signature)
                sentence += 1
            else:
                sentence += 1
        doc_count += 1
main()
# test_single()
