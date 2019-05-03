from collections import Counter
from pathlib import Path
from typing import Generator,List
import re

class ConllEntry:
    """
    Represents one entry in the CoNLL data setself.
    "_" is filled in for missing values.

    word_id    : Token counter, starting at 1 for each new sentence.
    form  : The form of the word.
    lemma : Lemma or stem of word form.
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
    #
    # def __str__(self):
    #     values = [str(self.id), self.form, self.lemma, self.cpos, self.pos, self.feats, str(self.pred_head) if self.pred_head is not None else None, self.pred_relation, self.deps, self.misc]
    #     return '\t'.join(['_' if v is None else v for v in values])


def vocab(conll_path):
    wordsCount = Counter()
    posCount = Counter()
    relCount = Counter()

    with open(conll_path, 'r') as conllFP:
        for sentence in read_conll(conllFP):
            wordsCount.update([node.norm for node in sentence if isinstance(node, ConllEntry)])
            posCount.update([node.pos for node in sentence if isinstance(node, ConllEntry)])
            relCount.update([node.relation for node in sentence if isinstance(node, ConllEntry)])

    return (wordsCount, {w: i for i, w in enumerate(wordsCount.keys())}, posCount.keys(), relCount.keys())


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
                if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                    entry_list.append(line.strip())
                else:
                    word_id = int(tok[0])
                    form    = tok[1]
                    cpos = tok[3]
                    pos = tok[4]
                    cpos = tok[5]
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

numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")
def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()

# def write_conll(fn, conll_gen):
#     with open(fn, 'w') as fh:
#         for sentence in conll_gen:
#             for entry in sentence[1:]:
#                 fh.write(str(entry) + '\n')
#             fh.write('\n')

if __name__ == '__main__':
    test_path = Path("data","en-universal-test.conll")
    conll_generator = read_conll(test_path)
    for sentence in conll_generator:
        # show word and it's head
        for entry in sentence:
            print(entry.form)
            print(entry.head)
        break
