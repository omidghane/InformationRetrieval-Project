
from parsivar import *

# Transform a text into a standard form.

import json

import re

import structlinks

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

def check_docID_exist(postings_lst, docs_index):
  if postings_lst[-1][0] == docs_index:
      return True

from structlinks import LinkedList

def creating_posting_list(all_docs):
  posting_map = HashMap(30)

  docs_index = -1
  tokens_index = -1
  for doc in all_docs:
    docs_index += 1

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

              if len(postings_lst) > docs_index:
                  postings_lst.insert(docs_index, (docs_index, 1, lst_positions))
              else:
                  postings_lst.append((docs_index, 1, lst_positions))

          # update postings_lst 
          else:
              docID, freq, positions_lst = postings_lst[len(postings_lst)-1]
              freq += 1
              positions_lst.append(tokens_index)
              postings_lst[len(postings_lst)-1] = (docID, freq, positions_lst)
            
          posting_map.put(token, (token_freq, postings_lst))
  # print(posting_map)
  return posting_map

def print_postings(posting_lists_map):
  for term in posting_lists_map.keys():
    print(term, end=" ----> ")
    freq, postings = posting_lists_map.get(term)
    print(freq , " :" )
    for posting in postings:
      docID, freq, positions = posting
      print(docID, " (", freq,"): ", positions)
    print("=====================")

def tokenizing(data, finish_index):
    all_tokens = []
    doc_topic_URL_map = HashMap(100000)

    for key in data.keys():
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

            finish_index += 1
            if finish_index == 500:
              break
        else:
            print('Error: "content" key not found in JSON file')

    return all_tokens, doc_topic_URL_map

once = True
with open('/content/drive/MyDrive/RI_file/IR_data_news_12k.json', 'r') as file:
    data = json.load(file)
tokenizer = Tokenizer()

doc_topic_URL_map = HashMap(100000)
all_tokens = []
finish_index = 0

all_tokens, doc_topic_URL_map = tokenizing(data, finish_index)

posting_lists = creating_posting_list(all_tokens)
# print_postings(posting_lists)

def checkPositions(e1, e2):
    # pdb.set_trace()

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
def PostingsList_intersect(p1, p2, pri_q, query_length, priority, flag, removed_list_tokens):

    answer = LinkedList.LinkedList()

    i1 = 0
    i2 = 0   # Indices to the elements in the postings lists
    while  i1 < len(p1) and i2 < len(p2):
        e1 = p1[i1]
        e2 = p2[i2]

        if  e1[0] == e2[0] :
            # pdb.set_trace()
            print(e1[0]," - ", e1[2], " same doc1")
            print(e2[0]," - ", e2[2], " same doc2")
            if flag == 3:
                ans_list = checkPositions(e1, e2)
                if ans_list:
                  answer.append((e1[0], e1[1], ans_list))
                else:
                  i1 += 1
                  i2 += 1
                  continue

            if flag == 2:
                pri_q.put(( (query_length-priority), e1[0]))
                print(( (query_length-priority), e1[0]), "removing pri_q")
                removed_list_tokens.append(e1[0])

            if flag == 1:
                answer.append((e1[0], e1[1], e1[2]))
            i1 += 1
            i2 += 1
        elif  e1[0] < e2[0]:
            if flag != 3:
              print((query_length-priority) , " - ", e1[0], " y2")
              pri_q.put(( (query_length-priority), e1[0]))
            i1 += 1
        else:
            if flag != 3:
              print( query_length, " - ", e2[0], " y3")
              pri_q.put((query_length, e2[0]))
            i2 += 1

    # print(answer.keys())
    while i1 < len(p1):
        e1 = p1[i1]
        i1 += 1
    while i2 < len(p2):
        e2 = p2[i2]
        i2 += 1

    

    return answer, removed_list_tokens

from queue import PriorityQueue
def searchQuery(query, posting_lists):
    flag = 1
  
    query_tokens = tokenizer.tokenize_words(query)
    query_length = len(query_tokens)
    pri_q = PriorityQueue()
    removed_list_tokens = []

    # print(query_tokens)
    # for token in query_tokens:
    #     postings = posting_lists.get(token)[1]
    #     print(token , end=" ")
    #     for posting in postings:
    #         print("[", posting[0], " - ", posting[2], end=" ] ")
    #     print()

    termID = 0
    old_answer = posting_lists.get(query_tokens[termID])[1]
    print(old_answer, " old one")
    # if query_length == 1:
    #     for t in old_answer:
    #         answer.append(t)

    while termID <= query_length-2:
        token = query_tokens[termID+1]
        # print(posting_lists.get(query_tokens[termID]), " old_answer ", token , " token")

        start_quote1 = query_tokens[termID].find('"')
        start_quote2 = query_tokens[termID+1].find('"')
        if token == "!":
            termID += 1
            token = query_tokens[termID+1]
            flag = 2
        
        if start_quote1 == -1 and start_quote2 == -1:
            p = posting_lists.get(token)[1]
            # print(p, " t1")
            answer, removed_list_tokens = PostingsList_intersect(old_answer , p, pri_q, query_length+1, termID, flag, removed_list_tokens)
            old_answer = answer
        # proccessing phrase query
        else:
            # pdb.set_trace()
            if start_quote1 == -1 and start_quote2 != -1:
                old_answer = posting_lists.get(token)[1]
            start_quote = query.find('"')
            end_quote = query.rfind('"')
            # Extract the part of the input between double quotes and store it in q2 variable
            phrase = query[start_quote :end_quote]
            # cleaned_phrase = [token.replace('"', '') for token in phrase]
            cleaned_phrase = phrase.replace('"', '')
            cleaned_tokens = tokenizer.tokenize_words(cleaned_phrase)
            
            print(cleaned_tokens)
            flag = 3
            index = 0
            old_answer2 = posting_lists.get(cleaned_tokens[index])[1]
            while index < len(cleaned_tokens)-1:
                token = cleaned_tokens[index+1]
                p1 = posting_lists.get(token)[1]
                
                print(old_answer2, " old")
                answer, removed_list_tokens = PostingsList_intersect(old_answer2 , p1, pri_q, query_length+1, termID, flag, removed_list_tokens)
                old_answer2 = answer
 
                index += 1

            termID += len(cleaned_tokens)-1
            # print(len(cleaned_phrase)-1, " termID")
            # print(len(query), " ql")

        print(answer)
        print("**************")

        termID += 1
        flag = 1

    for doc, freq, positions in answer:
        if doc not in removed_list_tokens:
          print(1, " - ", doc, " y1")
          pri_q.put((1, doc))

    while not pri_q.empty():
        next_item = pri_q.get()
        print(next_item)

end = False
while not end:
    # query = "باشگاه فوتسال ! آسیا"
    # طلا "لیگ برتر" ! والیبال
    query = "سهمیه المپیک"
    user_input = input("Enter your input: ")

    searchQuery(user_input, posting_lists)
    if input() == "end":
      break

