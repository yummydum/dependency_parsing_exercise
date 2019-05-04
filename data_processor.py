from collections import Counter
from pathlib import Path
from typing import Generator,List,Dict
import re
import torch

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

def read_conll(conll_path:Path) -> Generator[List[str],None,None]:
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
    wordsCount = Counter()
    posCount   = Counter()
    relCount   = Counter()

    conll_gen = read_conll(conll_path)
    for sentence in conll_gen:
        words    = []
        pos_tags = []
        relations= []
        # collect count for this sentence
        for node in sentence:
            words.append(node.norm)
            pos_tags.append(node.pos)
            relations.append(node.relation)
        # update counter
        wordsCount.update([node.norm for node in sentence])
        posCount.update([node.pos for node in sentence])
        relCount.update([node.relation for node in sentence])

    return wordsCount,posCount,relCount

def get_indexers(conll_path:Path) -> Tuple[Dict[str,int],Dict[str,int]]:
    """ Return Dictionaries which maps word/pos to an unique index."""
    words_count,pos_count,rel_count = count_word_stat(conll_path)
    word2index = {w: i for i, w in enumerate(wordsCount.keys())}
    pos2index  = {t: i for i, t in enumerate(posCount.keys())}
    return word2index,pos2index

def prepare_sequence(entry:ConllEntry) -> Tuple[]:
    idxs = [to_ix[w] for w in seq]
     return torch.tensor(idxs, dtype=torch.long)

def data_set(conll_path:Path) -> Generator[tensor]:
    """ Generates """
    conll_gen = read_conll(conll_path)
    word2index,pos2index = get_indexers(conll_path)
    for sentence in conll_data
        word_index = [word2index[w] for node.norm in sentence]
        pos_index  = [pos2index[t]  for node.pos  in sentence]
        yield torch.Longtensor(word_index),
              torch.LongTensor(pos_index)

if __name__ == '__main__':
    test_path = Path("data","en-universal-test.conll")
    conll_generator = read_conll(test_path)
    for sentence in conll_generator:
        # show word and it's head
        for entry in sentence:
            print(entry.form)
            print(entry.head)
        break

    word2index,pos2index = get_indexers(test_path)
