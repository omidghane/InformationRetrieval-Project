from parsivar import Tokenizer, FindStems

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

once = True
with open('/content/drive/MyDrive/RI_file/IR_data_news_12k.json', 'r') as file:
    data = json.load(file)
tokenizer = Tokenizer()

doc_title_map = HashMap(300)
all_tokens = []
finish_index = 0
for key in data.keys():
    if 'content' in data[key]:
        content = data[key]['content'][:-15]
        tokens = tokenizer.tokenize_words(content)

        normalized_content = normalize_text(tokens)
        noStopWords_content = remove_stopwords(normalized_content)
        stemmed_content = stemming(noStopWords_content)
        all_tokens.append(stemmed_content)
    else:
        print('Error: "content" key not found in JSON file')

posting_lists = creating_posting_list(all_tokens)
print_postings(posting_lists)



