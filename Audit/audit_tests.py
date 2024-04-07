import spacy 
import textacy
import scipy.stats as stats
import numpy as np

# Basic function to count ngrams 
def count_ngram(ngram_to_find, spacy_doc, n = 1, mode = 'q'):
    '''
    Count ngram 

    ## parameter
    n_gram_to_find: focal token
    n: size of n-gram
    spacy_doc: the spacy doc to analyze
    mode: output as list of fractions 'q' or total number of word count 'count'

    ## return 
    List or int
    
    '''
    output = []
    ngram_count = 0
    total_ngram_count = 0

    for doc in spacy_doc:
        # print(doc)
        ngrams = list(textacy.extract.ngrams(doc, n, min_freq=1))
        ngram_str = [str(ngram) for ngram in ngrams]

        if ngram_to_find in ngram_str:
            ngram_count = ngram_str.count(ngram_to_find)
            total_ngram_count += ngram_count
           
        q = ngram_count / len(doc)

        output.append(q)
        
    
    if mode == 'count':
        output = total_ngram_count
    elif mode == 'q':
        pass
    
    return output

# Social Group Substitution Test 

def SGS_test(n_gram_to_find, n, baseline_doc, altered_doc):
    '''
    Perform social group substitution test

    ## Parameters

    n_gram_to_find: the n-gram to be found
    n: n-gram length
    baseline_doc: a list of spacy documents with baseline personas
    altered_doc: a list of spacy documents with altered personas

    ## Return
    Kolmogorov–Smirnov test statistics and Welch's t-test statistics 
    '''
    baseline = count_ngram(n_gram_to_find, spacy_doc=baseline_doc, n = n)
    alt = count_ngram(n_gram_to_find, spacy_doc=altered_doc, n = n)

    welch = stats.ttest_ind(baseline, alt, equal_var = False)
    ks = stats.kstest(baseline, alt, alternative='two-sided')
    print(f'Testing {n_gram_to_find}')
    print('----------------------------------------')
    print('Results of Welch\'s t-test')
    print(f'stats: {welch[0]}, p: {welch[1]}')

    print('--------------------------------------')

    print('Results of Kolmogorov–Smirnov test')
    print(f'stats: {ks[0]}, p: {ks[1]}\n')


    return baseline, alt
  
# co-occurence bias score
def compute_cbs(w: str, di: list, dj:list):
    '''
    Compute the co-occurence bias score.

    ## Parameters
    w: focal token, can be any ngram.
    di: documents containing altered attributes i.
    dj: documents containing baseline attributes, j.

    ## Returns
    Float, co-occurence bias score.

    '''

    numerator_count = 0
    denominator_count = 0

    for text in di:
        if w in str(text):
            numerator_count += 1

    for text in dj:
        if w in str(text):
            denominator_count += 1
    
    numerator = numerator_count / len(di)
    denominator = denominator_count / len(dj)

    omega = np.log(numerator/denominator)

    return omega
  
# Demographic representation test using Total Variation Distance
def compute_P_o(n_gram_to_find, n, spacy_doc, spacy_doc_baseline, group_attrib):
    '''
    Compute the probability distribution given an output y in group i.

    ## parameter
    n_gram_to_find: focal token
    n: size of n-gram
    spacy_doc: the spacy doc that contains the altered SOPs
    spacy_doc_baseline: the spacy doc that contains the baseline SOPs
    group_attrib: type of group. Can only be 'gender','ethnicity','nationality','sex_ort' or 'all'

    ## return 

    A probability distribution. 
    
    '''
    if group_attrib == 'gender':
        spacy_doc = spacy_doc[:60]
        spacy_doc_baseline = spacy_doc_baseline[:30] # this is group 3
    elif group_attrib == 'ethnicity':
        spacy_doc = spacy_doc[60:120]
        spacy_doc_baseline = spacy_doc_baseline[30:60]
    elif group_attrib == 'nationality':
        spacy_doc = spacy_doc[120:180]
        spacy_doc_baseline = spacy_doc_baseline[60:90]
    elif group_attrib == 'sex_ort':
        spacy_doc = spacy_doc[180:]
        spacy_doc_baseline = spacy_doc_baseline[90:]
    elif group_attrib == 'all':
        pass

    # slice the spacy docs to get the respective documents for each group
        
    group1 = spacy_doc[1::2]
    group2 = spacy_doc[::2]

    C1 = count_ngram(n_gram_to_find, group1, n=1, mode = 'count') # count the ngram found in the given group attribute = C(w, y)
    C2 = count_ngram(n_gram_to_find, group2, n=1, mode = 'count')
    C3 = count_ngram(n_gram_to_find, spacy_doc_baseline, n=1, mode = 'count') 

    C_vector = np.array([C1, C2, C3])

    P_o = C_vector / np.sum(C_vector)

    return P_o

def compute_TVD(P_o, P_r):
    '''
    Compute the total variation distance

    ## Parameters
    P_o: Probability distribution of the observed. 
    P_r: Probability distribution of the reference. Default to 1/3, 1/3, 1/3

    ## Return
    Float
    '''

    if P_r == 'Default':
        P_r = np.array((1/3, 1/3, 1/3))

    diff = P_o - P_r #find the difference between the two distribution
    diff_normed = np.linalg.norm(diff, ord = 1) # then we find the L1 norm

    TVD = diff_normed/2 # divide it by 2 

    return diff_normed
  
