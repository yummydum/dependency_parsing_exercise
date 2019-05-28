from collections import Counter
from pathlib import Path
from typing import Generator,List,Dict,Tuple,Optional
import re
import numpy as np
from torch import LongTensor
from torch.utils.data import Dataset
from util import set_logger

logger = set_logger(__name__)
np.random.seed(1)

class ConllEntry:
    """
    Represents one entry in the CoNLL data setself.
    "_" is filled in for missing values.

    word_id    : Token counter, starting at 1 for each new sentence.
    form  : The form of the word.
    pos   : POS tag of the word.
    cpos  : CPOS tag of the word.
    head  : Head word of the word
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
    with conll_path.open(mode="r",encoding="utf-8") as f:
        for line in f:
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
                if tok[6] != '_':
                    head = int(tok[6])
                else:  # if the word does not have head, it's head is ROOT
                    head = -1
                relation = tok[7]
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
    for sentence in conll_gen:
        words    = []
        pos_tags = []
        relations= []
        # collect count for this sentence
        for entry in sentence:
            words.append(entry.norm)
            pos_tags.append(entry.pos)
            relations.append(entry.relation)
        # update counter
        words_count.update([entry.norm for entry in sentence])
        pos_count.update([entry.pos for entry in sentence])
        rel_count.update([entry.relation for entry in sentence])

    return words_count,pos_count,rel_count

class ConllDataSet(Dataset):

    def __init__(self,conll_path:Path,
                 word2index:Optional[Dict[str,int]]=None,
                 pos2index:Optional[Dict[str,int]] =None,
                 word_dropout=False):
        self.path = conll_path
        self.word_dropout = word_dropout

        # Set dict to map word/pos to index
        if word2index is None and pos2index is None:
            logger.debug("Loading data for training...")
            words_count,pos_count,rel_count = count_word_stat(conll_path)
            self.words_count = words_count
            self.word2index = {w: i for i, w in enumerate(words_count.keys())}
            self.pos2index  = {t: i for i, t in enumerate(pos_count.keys())}

            if word_dropout:
                logger.debug("Word drop out enabled")

        elif word2index is not None and pos2index is not None:
            self.word2index = word2index
            self.pos2index  = pos2index
        else:
            raise ValueError("Provide both indexer or nothing")

        # Size of input (add 1 for unknown token)
        self.vocab_size = len(self.word2index.keys()) + 1
        self.pos_size   = len(self.pos2index.keys())  + 1


        # Preprocess sentences
        logger.debug("Now preprocessing data...")
        self.data = []
        for sentence in read_conll(conll_path):
            word_index = [self.map2index(self.word2index,entry.norm) for entry in sentence]
            pos_index  = [self.map2index(self.pos2index,entry.pos)   for entry in sentence]
            head       = [entry.head for entry in sentence]
            data_i     = LongTensor(word_index).view(1,-1),LongTensor(pos_index).view(1,-1),head
            self.data.append(data_i)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self,idx:int) -> Tuple[LongTensor,LongTensor,List[int]]:
        return  self.data[idx]

    def map2index(self,index_dict:Dict[str,int],token:str):
        if token in index_dict:
            if self.word_dropout:
                drop_out_prob = 0.25 / (self.words_count[token] + 0.25)
                is_drop = np.random.binomial(1,drop_out_prob)
                if is_drop == 1:
                    return len(index_dict.keys())  # Index for unknown token
                else:
                    return index_dict[token]
            else:
                return index_dict[token]
        else:
            return len(index_dict.keys())  # Index for unknown token

if __name__ == '__main__':
    # Test
    test_path = Path("data","en-universal-test.conll")
    conll_generator = read_conll(test_path)
    for sentence in conll_generator:
        # show word and it's head
        for entry in sentence:
            print(entry.form)
            print(entry.head)
        break
