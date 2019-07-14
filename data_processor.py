"""
For Hydrogen;
%load_ext autoreload
%autoreload 2
"""

import csv
from collections import Counter
from pathlib import Path
import pickle
from typing import Generator,List,Dict,Tuple,Optional
import re
import numpy as np
# import pandas as pd
from tqdm import tqdm
import torch
from torch import LongTensor
from torch.utils.data import Dataset,TensorDataset,DataLoader, RandomSampler
from torchtext import data,datasets
# from pytorch_pretrained_bert import BertTokenizer, BertModel
from util import set_logger

logger = set_logger(__name__)
np.random.seed(1)

class ConllEntry:
    """
    Represents one entry in the CoNLL data set
    "_" is filled in for missing values.

    word_id : Token counter, starting at 0 (ROOT) for each new sentence.
    form    : The form of the word.
    pos     : POS tag of the word.
    cpos    : Coarse POS tag tag of the word.
    head    : The index of head word of the word
    relation : The label of the dependency relation
    """
    def __init__(self, word_id, form, pos, cpos, head, relation):
        self.id = word_id
        self.form = form
        self.norm = normalize(form)
        self.cpos = cpos.upper()
        self.pos = pos.upper()
        self.head = head
        self.relation = relation

# Compile in advance
numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")
def normalize(word:str) -> str:
    return 'NUM' if numberRegex.match(word) else word.lower()

def read_conll(conll_path:Path) -> Generator[List[ConllEntry],None,None]:
    """
    Read the conll data sequentially and return a generetor of list of entry
    """
    # Entry which represents the ROOT
    root = ConllEntry(0,'*root*','ROOT-POS','ROOT-CPOS',-1,'rroot')
    entry_list = [root]
    # Loop and generate list of string representing the sentence
    with conll_path.open(mode="r",encoding="utf-8") as f: # f = conll_path.open(), f.close()
        for line in f:   # line = next(f)
            tok = line.strip().split('\t')
            # if empty line, yield the sentence and init the next sentence
            if not tok or line.strip() == '':
                if len(entry_list)>1:
                    yield entry_list
                entry_list = [root]
            # else continue constructing the entry list
            else:
                word_id = int(tok[0])
                form    = tok[1]
                cpos    = tok[3]
                pos     = tok[4]
                cpos    = tok[5]
                head    = int(tok[6])
                relation = tok[7]

                # handle empty string
                if form == "''" or form == "``":
                    form = "<quote>"

                # Append the new token to the result
                new_entry = ConllEntry(word_id, form, pos, cpos,head,relation)
                entry_list.append(new_entry)

        # End of loop, yield the last sentence
        if len(entry_list) > 1:
            yield entry_list

def count_word_stat(conll_path:Path) -> Tuple[Counter,Counter,Counter]:
    words_count = Counter()
    pos_count   = Counter()
    rel_count   = Counter()
    conll_gen = read_conll(conll_path)
    logger.debug("Now counting word stats...")
    for i,sentence in enumerate(conll_gen):
        # update counter
        words_count.update([entry.norm for entry in sentence])
        pos_count.update([entry.pos for entry in sentence])
        rel_count.update([entry.relation for entry in sentence])
    return words_count,pos_count,rel_count

def word_dropout(word,count):
    drop_out_prob = 0.25 / (count + 0.25)
    is_drop = bool(np.random.binomial(1,drop_out_prob))
    if is_drop:
        return "<unk>"
    else:
        return word

def make_data(dropout=True):
    """ Convert the conll data to tsv file which could be loaded by torchtext """
    # Train data
    train_path = Path("data","en-universal-train.conll")
    if word_dropout:
        train_tsv_path = Path("data","train_word_dropout.tsv")
        # Count word stat
        logger.debug("Word drop out enabled")
        logger.debug("Counting data for word dropout...")
        words_count,pos_count,rel_count = count_word_stat(train_path)
        # Write to tsv
        with train_tsv_path.open(mode="w") as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            tsv_writer.writerow(["text","pos","head"])
            for sentence in read_conll(train_path):
                word_list = []
                pos_list = []
                head_list = []
                for token in sentence: # sentence[4].norm
                    t = word_dropout(token.norm,words_count[token.norm])
                    word_list.append(t)
                    pos_list.append(token.pos)
                    head_list.append(str(token.head))

                tsv_writer.writerow([" ".join(word_list),
                                     " ".join(pos_list),
                                     " ".join(head_list)])
    else:
        raise NotImplementedError()

    # Dev/Train deta
    for typ in ["dev","test"]:
        source_path   = Path("data",f"en-universal-{typ}.conll")
        result_tsv_path = Path("data",f"{typ}.tsv")
        with result_tsv_path.open(mode="w") as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            tsv_writer.writerow(["text","pos","head"])
            for sentence in read_conll(source_path):  # sentence = train_data[0]
                word_list = []
                pos_list = []
                head_list = []
                for token in sentence: # sentence[4].norm
                    word_list.append(token.norm)
                    pos_list.append(token.pos)
                    head_list.append(str(token.head))

                tsv_writer.writerow([" ".join(word_list),
                                     " ".join(pos_list),
                                     " ".join(head_list)])

# manage device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# torchtext constant
TEXT = data.Field(sequential=True,include_lengths=True,batch_first=True)
POS  = data.Field(sequential=True,include_lengths=True,batch_first=True)
HEAD = data.Field(sequential=True,use_vocab=False,pad_token=np.nan,
                  preprocessing=lambda x: [int(i) for i in x],
                  dtype=torch.float,batch_first=True)
tsv_fld = {"text":("text",TEXT),"pos":("pos",POS),"head":("head",HEAD)}

def load_iterator(which_data):  # data_path = train_tsv_path

    """
    Return torchtext.data.Itereator.
    For head words, np.nan is used for padding.
    """

    # Process train data
    if which_data == "train":
        data_path = Path("data","train_word_dropout.tsv")
        dataset = data.TabularDataset(path=data_path,format="tsv",fields=tsv_fld)
        # Construct vocab
        TEXT.build_vocab(dataset)
        POS.build_vocab(dataset)
        return data.BucketIterator(dataset,batch_size=32,shuffle=True,
                                   sort_key=lambda x:len(x.text),
                                   sort_within_batch=True,
                                   device=device)

    elif which_data == "dev":
        data_path = Path("data","dev.tsv")
        dataset = data.TabularDataset(path=data_path,format="tsv",fields=tsv_fld)
        return data.BucketIterator(dataset,batch_size=32,shuffle=True,
                                   sort_key=lambda x:len(x.text),
                                   sort_within_batch=True,
                                   device=device)

    elif which_data == "test":
        data_path = Path("data","test.tsv")
        dataset = data.TabularDataset(path=data_path,format="tsv",fields=tsv_fld)
        # Init Iterator (DataLoader)
        return data.Iterator(dataset,batch_size=32,device=device)

if __name__ == '__main__':
    # make_data()
    trn = load_iterator("train")
    dev = load_iterator("dev")
    test = load_iterator("test")

    # # see content
    # batch = next(iter(trn)) batch.head
    # # Debug HEAD
    # batch_size = 32
        # fields = dataset.fields.keys()  # copy field names  dir(trn.dataset.fields)
    # batches = data.iterator.batch(trn.data(),32,None)
    # minibatch = next(iter(batches))
    # name = "head"
    # field = HEAD
    # batch = [getattr(x, name) for x in minibatch]
    # padded = field.pad(batch)
    # field.numericalize(padded)
    # field.process(batch, device=device)

    # Test
    # conll_path = Path("data","en-universal-train.conll")
    # conll_generator = read_conll(conll_path)
    # for i,sentence in enumerate(conll_generator):
    #     # show word and it's head
    #     for entry in sentence:
    #         print(entry.form)
    #         print(entry.head)
    #     if i == 10:
    #         break
    #     pass


    # create
    # from pytorch_pretrained_bert.tokenization import BertTokenizer
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    #
    # for which in ["train","validation","test"]: # which = "train"
    #     input_path = f"../../data/processed/sentiment_tweet_noise_cleansed/{which}.csv"
    #     output_dir = f"../../data/test/processed/sentiment_BERT/"
    #     create_BERT_input(input_path,output_dir,tokenizer,[0,2,4],max_seq_length=100,mode="test")

    # bert_input_dir = f"../../data/test/processed/sentiment_BERT/"
    # batch_size = 32
    # dataloader = load_BERT_input(bert_input_dir,batch_size)



    # class ConllDataSet_BERT(Dataset):
    #
    #     def __init__(self,conll_path:Path):
    #         self.path = conll_path
    #         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #
    #         # Preprocess sentences
    #         logger.debug("Now preprocessing data...")
    #         self.data = []
    #         for sentence in read_conll(conll_path):
    #             tokenized_sentence = [self.tokenizer(self.word2index,entry.norm) for entry in sentence]
    #             word_index  =  self.tokenizer.convert_tokens_to_ids(word_index).view(1,-1)
    #             segment_ids =  np.zeros(len(word_index)).tolist()
    #             head       = [entry.head for entry in sentence]
    #             self.data.append((word_index,segment_ids,head))
    #
    #     def __len__(self) -> int:
    #         return len(self.data)
    #
    #     def __getitem__(self,idx:int) -> Tuple[LongTensor,LongTensor,List[int]]:
    #         return  self.data[idx]
    #
    # def create_BERT_input(input_path,output_dir,tokenizer,labels,max_seq_length=512,mode="test"):
    #     """
    #     Function:
    #     Create a csv file which is a valid input for BERT model;
    #     * Tokenize using BERT tokenizer
    #     * Convert the token to index
    #     * Append tags ([CLS],[SEP])
    #     * Create sengment id and input mask
    #     * Pad the sequence
    #     Arg:
    #     input_path     : The csv file containing the text data.
    #                      First column is the label, second column is the text data.
    #     output_dir     : The path of directory where the outputs is saved.
    #     tokenizer      : The BERT tokenizer used for tokenization.
    #     max_seq_length : The maximum length of the input.
    #                      The upper limmit is 512 by construction of BERT model.
    #     """
    #
    #     # setup
    #     assert max_seq_length <= 512
    #     logger = set_logger()
    #     input_path = Path(input_path)
    #     if mode == "test":
    #         logger.debug("TEST mode: the script will stop after processing 100 data")
    #
    #     # map label to index
    #     label2index = dict()
    #     for index,label in enumerate(labels):
    #         label2index[label] = index
    #
    #     # result path and accumulators
    #     contents = {"input_idxs","input_mask","segment_id","label"}
    #     writer_dict = {}
    #     for c in contents:
    #         result_path = Path(output_dir,f"{c}.csv")
    #         writer_dict[c] = csv.writer(result_path.open("w"),lineterminator="\n")
    #
    #     with input_path.open(mode="r") as f:
    #         original_data = csv.reader(f)
    #         next(original_data)
    #         for count, row in tqdm(enumerate(original_data)):
    #             text = row[0]
    #             label_index = label2index[int(row[1])]
    #             tokens = tokenizer.tokenize(text)
    #
    #             # truncate to maximum length
    #             # Account for [CLS] and [SEP] with "- 2"
    #             if len(tokens) > max_seq_length - 2:
    #                 tokens = tokens[:(max_seq_length - 2)]
    #
    #             # append tag
    #             tokens = ["[CLS]"] + tokens + ["[SEP]"]
    #
    #             # convert the tokens to index
    #             input_idxs = tokenizer.convert_tokens_to_ids(tokens)
    #
    #             # The mask has 1 for real tokens and 0 for padding tokens
    #             # Zero-pad up to the sequence length
    #             input_mask = [1] * len(input_idxs)
    #             padding = [0] * (max_seq_length - len(input_idxs))
    #             input_idxs += padding
    #             input_mask += padding
    #
    #             # create segment id (currently single sentence is assumed as input)
    #             segment_id = [0] * len(input_idxs)
    #
    #             assert len(input_idxs) == max_seq_length
    #             assert len(input_mask) == max_seq_length
    #             assert len(segment_id) == max_seq_length
    #
    #             writer_dict["input_idxs"].writerow(input_idxs)
    #             writer_dict["input_mask"].writerow(input_mask)
    #             writer_dict["segment_id"].writerow(segment_id)
    #             writer_dict["label"].writerow([label_index])
    #
    #             # show the log for the first 5 data
    #             if count < 5:
    #                 logger.debug(f"tokens:      {' '.join([str(x) for x in tokens])}")
    #                 logger.debug(f"input_idxs:  {' '.join([str(x) for x in input_idxs])}")
    #                 logger.debug(f"input_mask:  {' '.join([str(x) for x in input_mask])}")
    #                 logger.debug(f"segment_id: {' '.join([str(x) for x in segment_id])}")
    #
    #             if mode == "test" and count == 99:
    #                 logger.debug("Terminate: test mode finished")
    #                 break
    #
    # def load_BERT_input(bert_input_dir,batch_size=32):
    #     """ Load BERT input and wrap by DataLoader"""
    #
    #     logger = set_logger()
    #
    #     contents = ["input_idxs","input_mask","segment_id","label"]
    #     input_tensors = []
    #     for c in contents:
    #         logger.debug(f"loading {c}...")
    #         bert_input_path = Path(bert_input_dir,f"{c}.csv")
    #         bert_input_tensor = torch.LongTensor(pd.read_csv(bert_input_path).values)
    #         input_tensors.append(bert_input_tensor)
    #
    #     logger.debug("wrapping the tensors into DataLoader...")
    #     data_set =  TensorDataset(*tuple(input_tensors))
    #     sampler = RandomSampler(data_set)
    #     data_loader = DataLoader(data_set,sampler=sampler,batch_size=batch_size)
    #     logger.debug("load finished")
    #     return data_loader
