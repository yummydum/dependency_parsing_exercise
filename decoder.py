# This file contains routines from Lisbon Machine Learning summer school.
# The code is freely distributed under a MIT license. https://github.com/LxMLS/lxmls-toolkit/

import numpy as np
from typing import List,Tuple

def eisner_decode(score_matrix:List[List[float]],gold:List[int]=None) -> List[int]:
    """
    Implementation of Eisner's algorithm.

    Inputs:
    score_matrix : (i,j) element stores the score of i being the head word of j.
    gold         : ith element stores the index of the correct head word of the ith word.

    Function:
    Find argmax(score + hamming cost) when gold is not None (train mode)
    Find argmax(score)                when gold is None     (test mode)

    Output:
    1) An numpy array where the ith element stores the index of the infered head word of the ith word.
    2) The maximum score obtained by the decoded sequence
    """

    score_matrix = np.array(score_matrix)
    row_num,col_num = score_matrix.shape
    assert row_num == col_num

    # Number of words in the sentence (exclude ROOT)
    N = row_num - 1

    # Initialize CKY table as 3 dimential array;
    # CKY[s,t,:] stores the maximum score for sentence[s:t]
    # The direction of the head word
    #   => 0 for left  tree
    #   => 1 for right tree
    complete   = np.zeros([N+1, N+1, 2])
    incomplete = np.zeros([N+1, N+1, 2])

    # Table which stores the back pointer
    complete_backtrack   = -np.ones([N+1, N+1, 2], dtype=int)
    incomplete_backtrack = -np.ones([N+1, N+1, 2], dtype=int)

    # Fill -inf for cell which does not exist
    incomplete[0,:,0] -= np.inf

    # k: length of the span            (k=1,2,...,N)
    # s: where the segmentation starts (s=0,1,...,N-k)
    for k in range(1,N+1):
        for s in range(N-k+1):
            t = s+k

            # Incomplete span
            ## Find the maximum score
            incomplete_vals =  complete[(s+1):(t+1), t, 0] + complete[s, s:t, 1]
            max_score = np.max(incomplete_vals)
            back_track = s + np.argmax(incomplete_vals)
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
            complete[s, t, 0] = np.max(complete_vals0)
            complete_backtrack[s, t, 0] = s + np.argmax(complete_vals0)
            ## right tree
            complete_vals1 = incomplete[s, (s+1):(t+1), 1] + complete[(s+1):(t+1), t, 1]
            complete[s, t, 1] = np.max(complete_vals1)
            complete_backtrack[s, t, 1] = s + 1 + np.argmax(complete_vals1)

    heads = -np.ones(N+1, dtype=int)
    backtrack_eisner(incomplete_backtrack, complete_backtrack, 0, N, 1, 1, heads)
    return heads.tolist()

def backtrack_eisner(incomplete_backtrack, complete_backtrack, s, t, direction, complete, heads):
    '''
    Backtracking step in Eisner's algorithm.
    - incomplete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
    an end position, and a direction flag (0 means left, 1 means right). This array contains
    the arg-maxes of each step in the Eisner algorithm when building *incomplete* spans.
    - complete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
    an end position, and a direction flag (0 means left, 1 means right). This array contains
    the arg-maxes of each step in the Eisner algorithm when building *complete* spans.
    - s is the current start of the span
    - t is the current end of the span
    - direction is 0 (left attachment) or 1 (right attachment)
    - complete is 1 if the current span is complete, and 0 otherwise
    - heads is a (NW+1)-sized numpy array of integers which is a placeholder for storing the
    head of each word.
    '''
    if s == t:
        return
    if complete:
        r = complete_backtrack[s][t][direction]
        if direction == 0:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 0, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 0, 0, heads)
            return
        else:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 0, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 1, 1, heads)
            return
    else:
        r = incomplete_backtrack[s][t][direction]
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

    # test correctness of the algorithm
    # ROOT a b c

    # test case 1
    # ROOT -> a -> b -> c
    score_matrix = np.array([[0,10,0,0],
                             [0,0,10,0],
                             [0,0,0,10],
                             [0,0,0,0]])
    expected_heads = [-1,0,1,2]
    heads = eisner_decode(score_matrix)
    assert (expected_heads == heads).all()

    # test case 2
    # ROOT -> b
    # b -> a
    # b -> c
    score_matrix = np.array([[0,0,4,0],
                             [0,0,0,0],
                             [4,0,0,8],
                             [0,0,0,0]])
    expected_heads = [-1,2,0,2]
    heads = eisner_decode(score_matrix)
    assert (expected_heads == heads).all()

    # test case 3
    # ROOT -> c -> b -> a
    score_matrix = np.array([[0,0,0,100],
                             [0,0,0,0],
                             [0,100,0,0],
                             [0,0,100,0]])
    expected_heads = [-1,2,3,0]
    heads = eisner_decode(score_matrix)
    assert (expected_heads == heads).all()
