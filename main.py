
# %%
import collections.abc
#hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
#Now import hyper
import hyper

# %%
from collections.abc import Iterable
import pdb
from parsivar import *

# Transform a text into a standard form.

import json

import re

from queue import PriorityQueue

import structlinks


# %%

class HashMap:
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.size = 0
        self.buckets = [None] * self.capacity

    def hash(self, key):
        # Hashes the key and returns an index for the bucket
        return hash(key) % self.capacity

    def put(self, key, value):
        # Adds a key-value pair to the map
        index = self.hash(key)
        if self.buckets[index] is None:
            self.buckets[index] = []
        for pair in self.buckets[index]:
            if pair[0] == key:
                pair[1] = value
                return
        self.buckets[index].append([key, value])
        self.size += 1

    def get(self, key):
        # Returns the value associated with the given key
        index = self.hash(key)
        if self.buckets[index] is None:
            # there is no such key
            return "not"
        for pair in self.buckets[index]:
            if pair[0] == key:
                return pair[1]
        # raise KeyError(key)
        return "not"

    def keys(self):
        # Returns a list of all keys in the map
        keys = []
        for bucket in self.buckets:
            if bucket is not None:
                for pair in bucket:
                    keys.append(pair[0])
        return keys

    def __len__(self):
        # Returns the number of key-value pairs in the map
        return self.size

def stemming(tokens):
    # Initialize stemmer
    stemmer = FindStems()

    # Stem each token using the stemmer
    stemmed_tokens = [stemmer.convert_to_stem(token) for token in tokens]

    return stemmed_tokens


def remove_stopwords(tokens):
    # Define a list of stopwords
    persian_stopwords = ['و', 'در', 'به', 'از', 'که', 'این', 'را', 'با', 'های', 'برای', 'یک', 'شود', 'هر', 'شده', 'ای']

    # Remove stopwords from the list of tokens
    filtered_tokens = [token for token in tokens if token.lower() not in persian_stopwords]

    return filtered_tokens

def normalize_text(tokens):
    normalizer = Normalizer()
    # Convert the list of tokens into a string
    text = ' '.join(tokens)

    # Replace Arabic and Persian digits with English digits
    text = text.replace('۰', '0').replace('۱', '1').replace('۲', '2').replace('۳', '3').replace('۴', '4') \
        .replace('۵', '5').replace('۶', '6').replace('۷', '7').replace('۸', '8').replace('۹', '9')

    # Normalize the text
    text = normalizer.normalize(text)

    # Remove non-word characters
    text = re.sub(r'[^\w\s]', '', text)

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Convert all characters to lowercase
    text = text.lower()

    tokens = tokenizer.tokenize_words(text)

    return tokens

def check_docID_exist(posting_lst, docs_index):
    if posting_lst[-1][0] == docs_index:
        return True

from structlinks import LinkedList

def print_postings(posting_lists_map):
  for term in posting_lists_map.keys():
    print(term, end=" ----> ")
    freq, postings = posting_lists_map.get(term)
    print(freq , " :" )
    for posting in postings:
      docID, freq, positions = posting
      print(docID, " (", freq,"): ", positions)
    print("=====================")


# %%
def tokenizing(data, finish_index):
    all_tokens = []
    doc_topic_URL_map = HashMap(100000)

    for key in data.keys():
        print(key)
        if 'content' in data[key]:

            content = data[key]['content'][:-15]
            tokens = tokenizer.tokenize_words(content)

            normalized_content = normalize_text(tokens)
            noStopWords_content = remove_stopwords(normalized_content)
            stemmed_content = stemming(noStopWords_content)
            all_tokens.append(stemmed_content)

            # print(key,(data[key]['url']," - ",data[key]['title']), " this")
            doc_topic_URL_map.put(key,(data[key]['url'],data[key]['title']))

            # if once:
            #   print(tokens)
            #   print(stemmed_content)
            #   once = False

            # finish_index += 1
            # if finish_index == 1000:
            #   break
        else:
            print('Error: "content" key not found in JSON file')

    return all_tokens, doc_topic_URL_map

once = True
with open('IR_data_news_5k.json', 'r') as file:
    data = json.load(file)
tokenizer = Tokenizer()

doc_topic_URL_map = HashMap(100000)
all_tokens = []
finish_index = 0

all_tokens, doc_topic_URL_map = tokenizing(data, finish_index)

"""**bold text**"""


# print_postings(posting_lists)

# merging postions lists


# %%
def creating_posting_list(all_docs):
  # print("nuck")
  posting_map = HashMap(30)
  doc_checked = []

  docs_index = -1
  tokens_index = -1
  for doc in all_docs:
    print(len(doc_checked))
    docs_index += 1
    doc_checked.append(doc)

    for token in doc:
      tokens_index += 1

      # initiate posting_map
      if posting_map.get(token) == "not":
          # Linked_List (docID, frequency_in_doc, LinkedList)
          lst_postings = LinkedList.LinkedList()
          # Linked_List of positions of term like "1-> 4 -> 7"
          lst_positions = LinkedList.LinkedList()

          lst_positions.append(tokens_index)
          lst_postings.append((docs_index, 1, lst_positions))
          posting_map.put(token, (1, lst_postings))
      # update posting_map
      else:
          token_freq, postings_lst = posting_map.get(token)
          token_freq += 1

          # initiate postings_lst
          if not check_docID_exist(postings_lst, docs_index):
              lst_positions = LinkedList.LinkedList()
              lst_positions.append(tokens_index)

            #   print( token," - ", (docs_index, 1, lst_positions) , " ooooo")
              # print(postings_lst, " bbbbb")
              if len(postings_lst) > docs_index:
                  postings_lst.insert(docs_index, (docs_index, 1, lst_positions))
              else:
                  postings_lst.append((docs_index, 1, lst_positions))
              # print(postings_lst, " aaaaaaaaaaaafter")

          # update postings_lst
          else:
              docID, freq, positions_lst = postings_lst[len(postings_lst)-1]
              freq += 1
              positions_lst.append(tokens_index)
              postings_lst[len(postings_lst)-1] = (docID, freq, positions_lst)

          posting_map.put(token, (token_freq, postings_lst))
  # print(posting_map)
  return posting_map

posting_lists = creating_posting_list(all_tokens)

# %%
def cal_champion_length(champion_list):
    length = 0
    temp_queue = PriorityQueue()

    while not champion_list.empty():
        item = champion_list.get()
        temp_queue.put(item)
        length += 1

    # Restore the original PriorityQueue
    while not temp_queue.empty():
        item = temp_queue.get()
        champion_list.put(item)
    return length

def remove_from_champion(item_to_remove, championList):
    remove_item = (-item_to_remove[1], item_to_remove)
    # Create a temporary list to store items
    temp_list = []

    # Iterate over items in PriorityQueue and add to the temporary list, excluding the item to remove
    while not championList.empty():
        item = championList.get()
        # print(remove_item, " cccccc")
        if item != remove_item:
            # print(item, " okkkkkkkkkk")
            temp_list.append(item)
        else:
            pass
            # print(item, " removedddddddddd")

    # Create a new PriorityQueue and add the items from the temporary list
    # new_pri_queue = PriorityQueue()
    for item in temp_list:
        championList.put(item)

    # while not championList.empty():
    #     item = championList.get()
    #     # print(item, " dddddd")
    #     temp_list.append(item)

    # # Create a new PriorityQueue and add the items from the temporary list
    # # new_pri_queue = PriorityQueue()
    # for item in temp_list:
    #     championList.put(item)

    # return new_pri_queue

def check_docID_exist_champion(temp_list, docs_index):
    if temp_list[-1][0] == docs_index:
        return True
    # for pri , list in champion_list.queue:
    #     # print(list, " - ", docs_index, " pppppppp")
    #     if list[0] == docs_index:
    #         return True
    #     else:
    #         return False

def print_champList(championList, token):
    print("stert")
    for q in championList.queue:
        print(q, " - ", token)
    print("end")

def creating_champion_list(all_docs, K):
    # print("nuck")
    posting_map = HashMap(30)
    temp_map = HashMap(30)
    doc_checked = []

    docs_index = -1
    tokens_index = -1
    for doc in all_docs:
        print(len(docs_index))
        doc_checked.append(docs_index)
        docs_index += 1

        for token in doc:
            tokens_index += 1

            # initiate posting_map
            if posting_map.get(token) == "not":
                tmp_lst = LinkedList.LinkedList()

                # champion_list (-freq=priority , (docID, frequency_in_doc, LinkedList) )
                champion_list = PriorityQueue()
                # Linked_List of positions of term like "1-> 4 -> 7"
                lst_positions = LinkedList.LinkedList()

                lst_positions.append(tokens_index)
                champion_list.put( (-1,(docs_index, 1, lst_positions)) )
                posting_map.put(token, (1, champion_list))

                tmp_lst.append((docs_index, 1, lst_positions))
                temp_map.put(token, (1, tmp_lst))
                # print(tmp_lst, " ssssssss- ", token)
            # update posting_map
            else:
                token_freq, champion_list = posting_map.get(token)
                freq_tmp, temp_lst = temp_map.get(token)
                token_freq += 1

                # initiate positions_lst
                if not check_docID_exist_champion(temp_lst, docs_index):
                    lst_positions = LinkedList.LinkedList()
                    lst_positions.append(tokens_index)

                    # print( token," - ", (docs_index, 1, lst_positions) , " ooooo")
                    # remove_from_champion(temp_lst[len(temp_lst)-1], champion_list)
                    
                    champion_list.put( (-1, (docs_index, 1, lst_positions)) )
                    
                    temp_lst.append((docs_index, 1, lst_positions))
                    # print(temp_lst)

                # update postings_lst
                else:
                    docID, freq, positions_lst = temp_lst[len(temp_lst)-1]
                    freq += 1

                    # print_champList(champion_list, token)

                    positions_lst.append(tokens_index)
                    remove_from_champion(temp_lst[len(temp_lst)-1], champion_list)

                    # print_champList(champion_list, token)

                    champion_list.put( (-freq, (docID, freq, positions_lst)) )
                    temp_lst[len(temp_lst)-1] = (docID, freq, positions_lst)

                    # print_champList(champion_list, token)

                posting_map.put(token, (token_freq, champion_list))
                temp_map.put(token, (token_freq, temp_lst))

    res_map = HashMap(30)
    for term in posting_map.keys():
        index = 0
        token_freq, championLst = posting_map.get(term)
        newList = LinkedList.LinkedList()
        while not championLst.empty() and index < K:
            pri, item = championLst.get()
            newList.append(item)
            index += 1
        res_map.put(term, (token_freq, newList))

  # for term in posting_map:
  #     while not posting_map.get(term)[1].empty():
  #           priority, document = consine_priQueue.get()
  #           print(priority, " ", document, " resssss")

    return res_map

champion_lists = creating_champion_list(all_tokens, 100)

# %%
for term in posting_lists.keys():
      for i in posting_lists.get(term)[1]:
            print(i," - ", term)
      # print(champion_lists.get(term)[1])

# %%
import pickle
with open('data.pickle', 'wb') as file:
    pickle.dump(champion_lists, file)   

# %%
import pickle
import sys

# sys.setrecursionlimit(5001)

with open('data_posting.pickle', 'wb') as 
file:
    pickle.dump(posting_lists, file)

# %%
# Open the file in binary mode
with open('data.pickle', 'rb') as file:
    loaded_object = pickle.load(file)

# %%
# Open the file in binary mode
with open('data_posting.pickle', 'rb') as file:
    new_posting = pickle.load(file)

# %%
# making the term vectors
import math

def cal_number_duplicates(query_tokens):
    duplicates = []
    numberOfDuplicates = 0
    for q in query_tokens:
        if q in duplicates:
            numberOfDuplicates += 1
        duplicates.append(q)
    return numberOfDuplicates

def doc_vectors(vectors, query_tokens, posting_lists):
    # 11497 number of docs
    num_allDocs = 12200

    for doc in range(0, 2000):
        print(doc)
        index_term = 0
        duplicatedQueries = []
        for term in query_tokens:
            if term in duplicatedQueries:
                continue
            else:
                duplicatedQueries.append(term)
            freq_term_in_doc = 0

            index = 0
            current_node = posting_lists.get(term)[1][0]
            notFound = False
            while current_node is not None:
                
                if current_node[0] == doc:
                    # print(current_node, " index:", index, " doc:", doc)
                    freq_term_in_doc = current_node[1]
                    break
                elif current_node[0] > doc or index+1 >= len(posting_lists.get(term)[1]):
                    notFound = True
                    break
                index += 1
                current_node = posting_lists.get(term)[1][index]
            if notFound:
                val = 0
            else:
                val = (1+math.log(freq_term_in_doc)) * (math.log(num_allDocs)/len(posting_lists.get(term)[1]))
            # values += val
            # print(index_term, "-", val)
            vectors[doc][index_term] = val
            index_term += 1
        # print("doc:", doc, " val:", vectors[doc])

# query_tokens = tokenizer.tokenize_words("باشگاه باشگاه آسیا فوتبال آسیا ایران")
# query_tokens = stemming(query_tokens)
# numberOfDuplicates = cal_number_duplicates(query_tokens)
# vectors = [[0 for _ in range(len(query_tokens)-numberOfDuplicates)] for _ in range(1000)]
# doc_vectors(vectors, query_tokens)

# %%
from queue import PriorityQueue

def make_query_vector(queryVector):
    storedTerms = []
    index = 0
    for token in query_tokens:
        isDuplicated = False
        for e in storedTerms:
            if e[0] == token:
                isDuplicated = True
                queryVector[e[1]] += 1
        if isDuplicated:
            continue
        queryVector.append(1)
        storedTerms.append((token,index))
        index += 1

def consine_similarity(vectors, query_tokens):
    consine_priQueue = PriorityQueue()

    queryVector = []
    make_query_vector(queryVector)
    # print(queryVector)
    for doc in range(0, 12200):
        if vectors[doc] == [0,0,0,0]:
            continue
        soorat = 0
        doc_w = 0
        query_w = 0

        index_term = 0
        duplicatedQueries = []
        for term in query_tokens:
            if term in duplicatedQueries:
                continue
            else:
                duplicatedQueries.append(term)
            a = vectors[doc][index_term]
            b = queryVector[index_term]
            # print(a,"-",b," wwwwwwww")
            soorat += (a * b)
            doc_w += math.pow(a, 2)
            query_w += math.pow(b, 2)
            # print(soorat,"-",doc_w,"-",query_w, " qqqqq")
            index_term += 1
        makhraj = (doc_w ** 0.5) * (query_w ** 0.5)
        # print(soorat, "-", makhraj)
        if soorat == 0:
            similarity = 0
        else:
            similarity = soorat / makhraj
        consine_priQueue.put((-similarity, doc))
        
    return consine_priQueue

# vectors = [[0 for _ in range(len(query_tokens)-numberOfDuplicates)] for _ in range(1000)]
# query_tokens = tokenizer.tokenize_words("باشگاه باشگاه آسیا فوتبال آسیا ایران")
# query_tokens = stemming(query_tokens)
# doc_vectors(vectors, query_tokens)
# consine_similarity(vectors, query_tokens)

# %%
def checkPositions(e1, e2):

    lst1 = e1[2]
    lst2 = e2[2]
    new_list = LinkedList.LinkedList()

    i1 = 0
    i2 = 0
    while  i1 < len(lst1) and i2 < len(lst2):

        e1 = lst1[i1]
        e2 = lst2[i2]

        if e1+1 == e2:
            new_list.append(e2)
            i1 += 1
            i2 += 1
        elif e1 == e2 :
            i1 += 1
            i2 += 1
        elif e1 < e2:
            i1 += 1
        else:
            i2 += 1
    return new_list

# Intersect two postings lists that are sorted by docID
# if flag =3, then also merge the positions lists of tokens
# if flag =2, just add the same docs in removed_list_tokens to not add to answer list
def PostingsList_intersect(p1, p2, pri_q, query_length, priority, flag, removed_list_tokens):

    answer = LinkedList.LinkedList()

    if flag == 2:
        answer = p1

    i1 = 0
    i2 = 0   # Indices to the elements in the postings lists
    while  i1 < len(p1) and i2 < len(p2):
        e1 = p1[i1]
        e2 = p2[i2]

        if  e1[0] == e2[0] :
            if flag == 3:
                ans_list = checkPositions(e1, e2)
                if ans_list:
                  answer.append((e1[0], e1[1], ans_list))
                else:
                  i1 += 1
                  i2 += 1
                  continue

            if flag == 2:
                removed_list_tokens.append(e1[0])

            if flag == 1:
                answer.append((e1[0], e1[1], e1[2]))
            i1 += 1
            i2 += 1
        elif  e1[0] < e2[0]:
            # if flag == 1:
            #   pri_q.put((((query_length-priority)*1000)-e1[1], e1[0]))
            i1 += 1
        else:
            # if flag == 1:
            #   pri_q.put(((query_length*1000)-e2[1], e2[0]))
            i2 += 1
    #
    # while i1 < len(p1):
    #     e1 = p1[i1]
    #     i1 += 1
    # while i2 < len(p2):
    #     e2 = p2[i2]
    #     i2 += 1



    return answer, removed_list_tokens

# %%
from itertools import combinations
from queue import PriorityQueue


def searchQuery2(query_tokens, posting_lists):
    flag = 1
    item = LinkedList.LinkedList()


    query_length = len(query_tokens)
    pri_q = PriorityQueue()
    pri_qs = []

    removed_list_tokens = []


    compare_length = 2
    tID1 = -1
    num_combinations = len(list(combinations(query_tokens, 2)))
    old_answer = [[None] * (10) for _ in range(query_length+1)]
    index = 0

    if query_length == 1:
        i = 0
        t = posting_lists.get(query_tokens[0])[1]
        # for t in posting_lists.get(query_tokens[0])[1]:
        old_answer[1][i] = (t, list([0]))
            # i += 1
    # print(old_answer[1], " ppppppp")
    # print(posting_lists.get(query_tokens[0])[1], " gggggggg")

    while compare_length <= query_length:

        if index < num_combinations:
            tID1 += 1
        else:
            compare_length += 1
            num_combinations = len(list(combinations(query_tokens, compare_length)))
            print(num_combinations, " num of combinations", compare_length, " length")
            tID1 = 0
            index = 0

        flag = 1
        if query_tokens[tID1] == "!":
            tID1 += 1
            flag = 2
        token1 = query_tokens[tID1]

        for tID2 in range(0, query_length): # need_fix
            list_ids = []
            if compare_length == 2:
                p1 = posting_lists.get(token1)[1]
                list_ids = []
            else:
                p1, l = old_answer[compare_length-1][tID1]
                list_ids = l[:]
            # print(tID2, " tID2")
            if query_tokens[tID2] == "!":
                tID2 += 1
                flag = 2
            token2 = query_tokens[tID2]

            if tID2 in list_ids:
                # print(tID2, " is in list ids")
                continue
            if compare_length == 2 and tID1 == tID2:
                continue

            start_quote1 = query_tokens[tID1].find('"')
            start_quote2 = query_tokens[tID2].find('"')

            if (start_quote1 == -1) and start_quote2 == -1:
                # p1 = posting_lists.get(token1)[1]
                p2 = posting_lists.get(token2)[1]

                item, removed_list_tokens = PostingsList_intersect(p1 , p2, pri_q, query_length, tID2, flag, removed_list_tokens)
                # print(1)
                if compare_length == 2:
                    list_ids.append(tID1)
                    list_ids.append(tID2)
                else:
                    list_ids.append(tID2)

                old_answer[compare_length][index] = (item, list(list_ids))
                index += 1
                # print(index, " vvvvvvvvv")
    # print(old_answer[3][1])

    pri_qs.append(0)
    # pri_qs.append(0)
    for i in range(1, query_length+1):
        pri_q_new = PriorityQueue()
        added_documents = set()
        for item in old_answer[i]:
            if item is None:
                continue
            answer, terms = item
            for doc, freq, positions in answer:
                if doc not in removed_list_tokens and doc not in added_documents:
                    # print(doc, " ", terms, " ", freq, " ", (freq*(len(terms))))
                    pri_q_new.put((-freq, doc))
                    added_documents.add(doc)
        pri_qs.append(pri_q_new)
    
    # print(pri_q)

    return pri_qs

# query_tokens = tokenizer.tokenize_words("فوتبال")
# query_tokens = stemming(query_tokens)
# searchQuery2(query_tokens, champion_lists)

# %%
from queue import PriorityQueue
def searchQuery(query, posting_lists):
    flag = 1
    answer = LinkedList.LinkedList()

    query_tokens = tokenizer.tokenize_words(query)
    query_tokens = stemming(query_tokens)
    query_length = len(query_tokens)
    print(query_length, " query length")
    pri_q = PriorityQueue()
    removed_list_tokens = []

    termID = 0
    old_answer = posting_lists.get(query_tokens[termID])[1]
    # print(old_answer, " old_answer")

    # when we have a word query
    if query_length == 1:
        for t in old_answer:
            answer.append(t)

    while termID <= query_length-2:
        token = query_tokens[termID+1]

        start_quote1 = query_tokens[termID].find('"')
        start_quote2 = query_tokens[termID+1].find('"')

        # check if token is ! and set falg = 2
        if query_tokens[termID+1] == "!" or query_tokens[termID] == "!":
            if query_tokens[termID+1] == "!":
                termID += 1
            token = query_tokens[termID+1]
            flag = 2

        # proccessing token
        if start_quote1 == -1 and start_quote2 == -1:
            posting_new = posting_lists.get(token)[1]

            answer, removed_list_tokens = PostingsList_intersect(old_answer , posting_new, pri_q, query_length+1, termID, flag, removed_list_tokens)
            old_answer = answer

        # proccessing phrase query
        else:

            if start_quote1 == -1 and start_quote2 != -1:
                old_answer = posting_lists.get(query_tokens[termID])[1]
            start_quote = query.find('"')
            end_quote = query.rfind('"')

            # Extract the part of the input between double quotes and store it in q2 variable
            phrase = query[start_quote :end_quote]
            cleaned_phrase = phrase.replace('"', '')
            cleaned_tokens = tokenizer.tokenize_words(cleaned_phrase)

            flag = 3
            index = 0
            old_answer2 = posting_lists.get(cleaned_tokens[index])[1]
            while index < len(cleaned_tokens)-1:
                token = cleaned_tokens[index+1]
                p1 = posting_lists.get(token)[1]

                answer, igonre = PostingsList_intersect(old_answer2 , p1, pri_q, query_length+1, termID, flag, removed_list_tokens)
                old_answer2 = answer

                index += 1
            if start_quote1 == -1 and start_quote2 != -1:
                # print(old_answer2, " :old2", old_answer, " old here")
                flag = 1
                answer, igonre = PostingsList_intersect(old_answer2 , old_answer, pri_q, query_length+1, termID, flag, removed_list_tokens)
            old_answer = answer

            termID += len(cleaned_tokens)

        termID += 1
        flag = 1
    print(termID, "num of term qeuries")

    # putting other docs that are completelly satisfied to priority_queue
    for doc, freq, positions in answer:
        if doc not in removed_list_tokens:
          pri_q.put((12200-freq, doc))

    new_pri_q = PriorityQueue()
    while not pri_q.empty():
        pri, doc = pri_q.get()
        # if pri != query_length+1:
        new_pri_q.put((pri, doc))

    return new_pri_q

# %%
def print_output(q, user_input, K):
    # print("hereee")
    query_tokens = tokenizer.tokenize_words(user_input)
    query_tokens = stemming(query_tokens)
    index = 0
    # for pri_q in q:
    # if q == 0:
    #     continue
    while not q.empty():
        # print("KKKKKKK")
        if index > K:
            break
        priority, key = q.get()
        print("doc_number: ", key)
        sentences = tokenizer.tokenize_sentences(data[str(key)]['content'][:-15])
        print("title: ", doc_topic_URL_map.get(str(key))[1])
        print("URL: ", doc_topic_URL_map.get(str(key))[0])
        # print(sentences)
        for i in range(0, len(query_tokens)):
            token = query_tokens[i]
            start_quote = token.find('"')
            if token == "!":
                i += 1
                continue
            elif start_quote != -1:
                start_quote = user_input.find('"')
                end_quote = user_input.rfind('"')

                token = user_input[start_quote :end_quote]
                token = token.replace('"', '')
                # i += len(token)-1
                # continue
            for sentence in sentences:
                if token in sentence or token in tokenizer.tokenize_words(sentence):
                    print("- ", sentence)
        index += 1

# %%
def query_similarity_jakard(user_input, pri_qs):
    jakard_priQueue = PriorityQueue()
    # index = 1
    for index in range(len(pri_qs)-1, 0, -1):
        # print(index, " wwwwww")
        pri_q = pri_qs[index]
        if pri_q == 0:
            continue
        # index += 1
        while not pri_q.empty():
            priority, document = pri_q.get()
            # print(document, "-", len(all_tokens[document]), " eeeeeeee")
            score = index/(len(user_input)+len(all_tokens[document]))
            jakard_priQueue.put((-score, document))

    # while not jakard_priQueue.empty():
    #     priority, document = jakard_priQueue.get()
    #     print(priority, " ", document, " resssss")

    return jakard_priQueue


# %%

# end = False
# while not end:

user_input = input("Enter your input: ")
query_tokens = tokenizer.tokenize_words(user_input)
query_tokens = stemming(query_tokens)
numberOfDuplicates = cal_number_duplicates(query_tokens)

vectors = [[0 for _ in range(len(query_tokens)-numberOfDuplicates)] for _ in range(12200)]
# make doc vectors
doc_vectors(vectors, query_tokens, posting_lists)

# calculate consine similarity
consine_priQueue = consine_similarity(vectors, query_tokens)

# searching for the docs, contain user requirement
# q = searchQue
# ry(user_input, posting_lists)
pri_qs = searchQuery2(query_tokens, posting_lists)

# calculate jakard similarity
jakard_priQueue = query_similarity_jakard(query_tokens ,pri_qs)

# print doc and sentence containing user requirement
# print_output(jakard_priQueue, user_input, 5)
# print_output(consine_priQueue, user_input, 5)

    # if input() == "e":
    #   break

# %%
# print_output(jakard_priQueue, user_input, 5)
print_output(consine_priQueue, user_input, 5)

