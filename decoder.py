# This file contains routines from Lisbon Machine Learning summer school.
# The code is freely distributed under a MIT license. https://github.com/LxMLS/lxmls-toolkit/

import torch
from typing import List,Tuple

def eisner_decode(score_matrix:torch.tensor,gold:List[int]=None) -> Tuple[List[int],torch.tensor]:
    """
    Implementation of Eisner's algorithm.

    Inputs:
    score_matrix : (i,j) element stores the score of i being the head word of j.
    gold         : ith element stores the index of the correct head word of the ith word.

    Function:
    Find argmax(score + hamming cost) when gold is not None (train mode)
    Find argmax(score)                when gold is None     (test mode)

    Output:
    1) Array where the ith element stores the index of the infered head word of the ith word.
    2) The maximum score obtained by the decoded sequence
    """

    # N: number of the word in the sentence excluding the ROOT
    row_num,col_num = score_matrix.shape
    N = row_num - 1

    # Initialize CKY table as 3 dimential array;
    # CKY[s,t,:] stores the maximum score for sentence[s:t]
    # The direction of the head word
    #   => 0 for left  tree
    #   => 1 for right tree
    complete   = torch.zeros([N+1, N+1, 2])
    incomplete = torch.zeros([N+1, N+1, 2])

    # Table which stores the back pointer
    complete_backtrack   = -torch.ones([N+1, N+1, 2])
    incomplete_backtrack = -torch.ones([N+1, N+1, 2])

    # Fill -inf for cell which does not exist
    incomplete[0,:,0] -= float("nan")

    # k: length of the span            (k=1,2,...,N)
    # s: where the segmentation starts (s=0,1,...,N-k)
    for k in range(1,N+1):
        for s in range(N-k+1):
            t = s+k

            # Incomplete span
            ## Find the maximum score
            incomplete_vals =  complete[(s+1):(t+1), t, 0] + complete[s, s:t, 1]
            max_score = torch.max(incomplete_vals)
            back_track = s + torch.argmax(incomplete_vals)
            ## left tree (s <- t)
            hamming_cost = 0.0 if gold is not None and gold[s]==t else 1.0
            incomplete[s, t, 0] = max_score + score_matrix[t,s] + hamming_cost
            incomplete_backtrack[s, t, 0] = back_track
            ## right tree (s -> t)
            hamming_cost = 0.0 if gold is not None and gold[t]==s else 1.0
            incomplete[s, t, 1] = max_score + score_matrix[s,t] + hamming_cost
            incomplete_backtrack[s, t, 1] = back_track

            # Complete tree
            ## left tree
            complete_vals0 = complete[s, s:t, 0] + incomplete[s:t, t, 0]
            complete[s, t, 0] = torch.max(complete_vals0)
            complete_backtrack[s, t, 0] = s + torch.argmax(complete_vals0)
            ## right tree
            complete_vals1 = incomplete[s, (s+1):(t+1), 1] + complete[(s+1):(t+1), t, 1]
            complete[s, t, 1] = torch.max(complete_vals1)
            complete_backtrack[s, t, 1] = s + 1 + torch.argmax(complete_vals1)

    maximum_score = complete[0][N][1]
    heads = [-1 for _ in range(N+1)]
    backtrack_eisner(incomplete_backtrack,
                     complete_backtrack,
                     s=0,t=N,
                     direction=1,
                     complete =1,
                     heads=heads)

    return heads,maximum_score

def backtrack_eisner(incomplete_backtrack, complete_backtrack, s, t, direction, complete, heads):
    '''
    Backtracking step in Eisner's algorithm.
    - incomplete_backtrack is a (NW+1)-by-(NW+1) list indexed by a start position,
    an end position, and a direction flag (0 means left, 1 means right). This array contains
    the arg-maxes of each step in the Eisner algorithm when building *incomplete* spans.
    - complete_backtrack is a (NW+1)-by-(NW+1) list array indexed by a start position,
    an end position, and a direction flag (0 means left, 1 means right). This array contains
    the arg-maxes of each step in the Eisner algorithm when building *complete* spans.
    - s is the current start of the span
    - t is the current end of the span
    - direction is 0 (left attachment) or 1 (right attachment)
    - complete is 1 if the current span is complete, and 0 otherwise
    - heads is a (NW+1)-sized list of integers which is a placeholder for storing the
    head of each word.
    '''
    if s == t:
        return
    if complete:
        r = int(complete_backtrack[s][t][direction].item())
        if direction == 0:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 0, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 0, 0, heads)
            return
        else:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 0, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 1, 1, heads)
            return
    else:
        r = int(incomplete_backtrack[s][t][direction].item())
        if direction == 0:
            heads[s] = t
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r + 1, t, 0, 1, heads)
            return
        else:
            heads[t] = s
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r + 1, t, 0, 1, heads)
            return

if __name__ == '__main__':

    # Test correctness of the algorithm
    # [ROOT,a,b,c]
    # [0   ,1,2,3]

    # Test case 1
    # R -> a
    # a -> b
    # b -> c
    score_matrix = torch.tensor([[0,10,0,0],
                                 [0,0,10,0],
                                 [0,0,0,10],
                                 [0,0,0,0]])
    expected_heads = [-1,0,1,2]
    expected_score = 33
    heads,score = eisner_decode(score_matrix)
    assert expected_heads == heads
    assert expected_score == score.item()

    # Test case 2
    # R -> c
    # c -> a
    # c -> b
    score_matrix = torch.tensor([[0,0,0,10],
                                 [0,1,0,1],
                                 [0,2,0,1],
                                 [0,8,8,0]])
    expected_heads = [-1,3,3,0]
    expected_score = 29
    heads,score = eisner_decode(score_matrix)
    assert expected_heads == heads
    assert expected_score == score.item()

    # Check with actual data
    from pathlib import Path
    from data_processor import ConllDataSet

    dev_path = Path("data","en-universal-dev.conll")
    dev_data = ConllDataSet(dev_path)
    golden_head = dev_data[0][2]

    sentence_len = len(golden_head)
    score_matrix = torch.randn(sentence_len,sentence_len)
    for m,h in enumerate(golden_head):
        if m == 0:
            continue
        score_matrix[h][m] += 5
    head,score = eisner_decode(score_matrix)
