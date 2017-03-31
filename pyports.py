from math import log2
from collections import defaultdict
import json
import pickle
import sys
import argparse

RUS_LETTERS = 'абвгдеёжзийклмнопрстуфхчшщъыьэюя-'
ENG_LETTERS = 'abcdefghijklmnopqrstuvwxyz-'

class Tree(object):
    """
    A b-way tree of depth d, where b are the letters and d is the length of 
    the longest word. A path from the root to some node spells out the starting
    or ending fragment of some word. The node itself contains the count of the
    letter, at that depth. Once the tree has been normalized, the node also
    contains the probability of the letter out of any letter at this depth.
    """
    def __init__(self):
        self.count = 0
        self.prob = float('-inf')
        self.chain = float('-inf')
        self.subtrees = defaultdict(Tree)

    def from_corpus(self, corpus, backward=False):
        """
        Count up the number of letter transitions in the corpus.
        The corpus is an iterable of words (e.g. a list).

        Arguments:
            backward: read words from the end to build suffix trees,
                otherwise trees will be prefix trees (default: False)
        """
        for word in corpus:
            word = list(word.strip())
            self.increment(word, backward)
        self.log_normalize()
    
    def __repr__(self):
        return json.dumps(self._to_dict(), sort_keys=True, indent=4)

    def _to_dict(self):
        subtrees = {k:self.subtrees[k]._to_dict() for k in self.subtrees}
        return {'count': self.count, 'prob': self.prob, 'chain_prob': self.chain, 'subtrees': subtrees}
   
    def increment(self, word, backward=False):
        """
        Read the word (a list of characters), and recursively increment
        the transition counts for this tree and its subtrees.
        If backward is True, words are processed from the end of the word.
        """
        self.count += 1
        if not word:
            return #nothing to return since we're building up internal tree
        if not backward:
            letter = word.pop(0)
        else:
            letter = word.pop()
        self.subtrees[letter].increment(word, backward)

    def log_normalize(self, start=True):
        """
        Add probabilities and chain probabilities to the tree: normalize at
        each depth by the count of items at that depth, convert to log base 2.
        """
        if start: #initial node has prob of 1/vocab, self.count is vocab size
            self.prob = log2(1) - log2(self.count)
            self.chain = self.prob
        start = False
        for letter in self.subtrees:
            self.subtrees[letter].prob = log2(self.subtrees[letter].count) - log2(self.count)
            self.subtrees[letter].chain = self.chain + (log2(self.subtrees[letter].count) - log2(self.count))
            self.subtrees[letter].log_normalize(start)
    
class Segmenter(object):
    """
    An affix finder that must first be trained on a corpus of words, and can
    then be used to find prefixes and suffixes on new words.
    """
    def __init__(self):
        self.corpus = set()
        self.forward = Tree()
        self.backward = Tree()
        self.prefix_scores = defaultdict(int)
        self.suffix_scores = defaultdict(int)
        self.prefix_probs = {}
        self.suffix_probs = {}
        self.ppth = None
        self.spth = None
        self.pcth = None
        self.scth = None
   
    def __repr__(self):
        prefixes = json.dumps(self._to_dict(self.prefixes), sort_keys=True, indent=4)
        suffixes = json.dumps(self._to_dict(self.suffixes), sort_keys=True, indent=4)
        return 'Prefixes: {}\nSuffixes: {}'.format() 

    def _to_dict(self, d):
        return {k:d[k] for k in d}

    def save(self, fs):
        """Save the model to the file stream."""
        return pickle.dump(self, fs)

    def add_corpus(self, stream, letters=None, limit=None):
        """
        Return a set of types from the corpus, optionally filtered for words
        written in characters outside the alphabet of letters. 

        Arguments:
            stream: open file handle to read from  
            letters: exclude words with letters outside of these
            limit: the maximum allowed size of the corpus (default: None)
        """
        for word in stream:
            if limit and len(self.corpus) >= limit:
                break 
            word = self._normalize_word(word, letters=letters)
            if not word:
                continue 
            self.corpus.add(word)
        return self.corpus

    def train(self, ppth=0.6, spth=0.6, pcth=1.0, scth=1.0, verbose=True):
        """Train the affix finder using a corpus (list of words)."""
        if verbose: print('\nTraining on {} words with thresholds:'.format(len(self.corpus)))
        if verbose: print('  prefix parent threshold {}'.format(ppth))
        if verbose: print('  suffix parent threshold {}'.format(spth))
        if verbose: print('  prefix child threshold {}'.format(pcth))
        if verbose: print('  suffix child threshold {}'.format(scth))
        self.ppth = ppth
        self.spth = spth
        self.pcth = pcth
        self.scth = scth
        if verbose: print('Building trees...')
        self.forward.from_corpus(self.corpus, backward=False)
        self.backward.from_corpus(self.corpus, backward=True)
        if verbose: print('Scoring...')
        for word in self.corpus:
            self._score_prefixes(word, ppth, pcth)
            self._score_suffixes(word, spth, scth)
        if verbose: print('Cleaning prefixes...')
        self._clean(self.prefix_scores, self.prefix_probs)
        if verbose: print('Cleaning suffixes...')
        self._clean(self.suffix_scores, self.suffix_probs)
        if verbose: print('Pruning prefixes...')
        self._prune(self.prefix_scores, self.prefix_probs)
        if verbose: print('Pruning suffixes...')
        self._prune(self.suffix_scores, self.suffix_probs)
    
    def segment(self, word):
        """Return a correctly ordered list of morphemes in the word."""
        word = self._normalize_word(word)
        if not word:
            return []
        word, suffixes = self._find_suffixes(word)
        word, prefixes = self._find_prefixes(word)
        if prefixes is None:
            prefixes = []
        if word: #it's not necessary to add empty affixes
            prefixes.append(word) #just build off the prefixes list
        if suffixes is not None:
            prefixes.extend(suffixes)
        return prefixes

    def _find_suffixes(self, word, suffixes=None):
        """Find a list of suffixes on this word in the correct order."""
        if not suffixes:
            suffixes = []
        best_suffix = None
        lowest_prob = log2(1)
        for s in self.suffix_probs:
            p = self.suffix_probs[s]
            if word.endswith(s) and lowest_prob > p and p < log2(1):
                best_suffix = s
                lowest_prob = p
        if best_suffix is not None:
            suffixes.insert(0, best_suffix) #insert suffix for correct ordering
        else:
            return (word, suffixes)
        word = word[:-len(best_suffix)] #word up to the suffix
        word, suffixes = self._find_suffixes(word, suffixes)
        return (word, suffixes)
    
    def _find_prefixes(self, word, prefixes=None):
        """Find a list of prefixes on this word in the correct order."""
        if not prefixes:
            prefixes = []
        best_prefix = None
        lowest_prob = log2(1)
        for s in self.prefix_probs:
            p = self.prefix_probs[s]
            if word.startswith(s) and lowest_prob > p and p < log2(1):
                best_prefix = s
                lowest_prob = p
        if best_prefix is not None:
            prefixes.append(best_prefix) #append prefix for correct ordering
        else:
            return (word, prefixes)
        word = word[len(best_prefix):] #word starting after the prefix
        word, prefixes = self._find_prefixes(word, prefixes)
        return (word, prefixes)

    def _score_prefixes(self, word, pthresh, cthresh, i=None, tree=None):
        """
        Read forwards from the start of the word to score potential suffixes.
        The threshold is the minimum probability the target letter needs before
        being considered to be glued onto the current potential stem. 
        """
        if i is None:
            i = 0 
            tree = self.forward
        if i == len(word)-1: #no prefix to try after last letter 
            return
        
        prefix = word[:i+1]
        stem = word[i+1:]
        letter = word[i]
        tree = tree.subtrees[letter]
        cond1 = tree.prob >= log2(pthresh) #I'm glued to my parent
        cond2 = tree.subtrees[word[i+1]].prob < log2(cthresh) #my child can leave
        cond3 = stem in self.corpus
        all_cond = cond1 and cond2
        
        if all_cond:
            self.prefix_scores[prefix] += 19
            if prefix not in self.prefix_probs: #prevent extra loops later
                self.prefix_probs[prefix] = tree.chain
            #found one, start over from top of tree
            return self._score_prefixes(stem, pthresh, cthresh, 0, self.forward)
        else:
            self.prefix_scores[prefix] -= 1
            if prefix not in self.prefix_probs: #prevent extra loops later
                self.prefix_probs[prefix] = tree.chain
            i += 1
            return self._score_prefixes(word, pthresh, cthresh, i, tree)
            
    def _score_suffixes(self, word, pthresh, cthresh, i=None, tree=None):
        """
        Read backwards from the end of the word to score potential suffixes.
        The threshold is the minimum probability the target letter needs before
        being considered to be glued onto the current potential stem. 
        """
        if i is None:
            i = len(word)-1
            tree = self.backward
        if i == 0: #no suffix to try after first letter 
            return
        
        suffix = word[i:]
        stem = word[:i]
        letter = word[i]
        tree = tree.subtrees[letter]
        cond1 = tree.prob >= log2(pthresh) #I'm glued to my parent
        cond2 = tree.subtrees[word[i-1]].prob < log2(1) #my child can leave
        cond3 = stem in self.corpus
        all_cond = cond1 and cond2
        
        if all_cond:
            self.suffix_scores[suffix] += 19
            if suffix not in self.suffix_probs: #prevent extra loops later
                self.suffix_probs[suffix] = tree.chain
            #found one, start over from top of tree
            return self._score_suffixes(stem, pthresh, cthresh, len(stem)-1, self.backward)
        else:
            suffix = word[i:]
            self.suffix_scores[suffix] -= 1
            if suffix not in self.suffix_probs: #prevent extra loops later
                self.suffix_probs[suffix] = tree.chain
            i -= 1
            return self._score_suffixes(word, pthresh, cthresh, i, tree)

    def _clean(self, affix_scores, affix_probs):
        """Remove affixes that score below 0 (they're not real morphemes)."""
        remove = []
        for affix in affix_scores:
            if affix_scores[affix] < 0:
                remove.append(affix)
        if '' in affix_scores:
            affix_scores.pop('')
            affix_probs.pop('')
        for i in remove:  #do this at the end otherwise dict changes as you loop
            affix_scores.pop(i)
            affix_probs.pop(i)

    def _prune(self, affix_scores, affix_probs):
        """Remove morphemes that are concatenations of 2 higher score ones."""
        remove = set()
        for ai in affix_scores:
            for aj in affix_scores:
                af = ai+aj #the concatenated affix
                if af in affix_scores: 
                    if (affix_scores[af] < affix_scores[ai] and 
                        affix_scores[af] < affix_scores[aj]):
                        remove.add(af) 
        for i in remove: #do this at the end otherwise dict changes as you loop
            affix_scores.pop(i)
            affix_probs.pop(i)
   
    def _normalize_word(self, word, letters=None):
        """
        Lowercase the word. If letters are supplied, when there are unrecognized
        letters in the word, return an empty string instead of the word string.
        """
        word = word.strip().lower()
        if letters:
            for letter in word:
                if letter not in letters:
                    return ''
        return word

class Evaluator(object):
    def __init__(self):
        self.true_positives = None #gold has a break here and so do I
        self.false_positives = None #gold has a break here and I do not
        self.false_negatives = None #gold doesn't have a break here but I do
        self.gold_breaks = None #the number of morpheme boundaries in gold
        self.precision = None #true_positives / (true_positives+false_positives)
        self.recall = None #true_positives / (true_positives+false_negatives)
        self.fscore = None #https://en.wikipedia.org/wiki/F1_score

    def calculate(self, results, golds):
        """
        Calculate the precision, recall and fscore for the results as compared
        to the gold standard. 

        Arguments:
            results: a list of length n of induced word segmentation lists
            golds: a list of length n of correct word segmentation lists
                (the same words as in results)
        """
        self._count(results, golds)
        self._calculate_precision(self.true_positives, self.false_positives)
        self._calculate_recall(self.true_positives, self.false_negatives)
        self._calculate_fscore(self.precision, self.recall)
        return (self.precision, self.recall, self.fscore)
     
    def _glue(self, word, separator='+'):
        """Concatenate segments, using separator to denote morpheme breaks."""
        glued = ''
        for i, segment in enumerate(word):
            if i == 0:
                glued = segment
            else:
                glued += separator + segment
        return glued

    def _count(self, results, golds, separator='+'):
        """Count up the number of true pos, false pos, and false neg."""
        assert len(results) == len(golds), 'Results ({} entries) != gold ({} entries)'.format(len(results), len(gold)) 
        self.true_positives = 0 
        self.false_positives = 0
        self.false_negatives = 0
        for e, result in enumerate(results):
            gold = golds[e]
            ri = 0 #how far we are through the result word
            gi = 0 #how far we are through the gold word
            rj = self._glue(result, separator)
            gj = self._glue(gold, separator)
            while ri < len(''.join(gold)):
                rc = rj[ri]
                gc = gj[gi]
                if rc == separator:           #result has morpheme break here..
                    if gc == separator:       #...and so does gold
                        self.true_positives += 1
                        ri += 1
                        gi += 1
                    else:                     #...but gold doesn't
                        self.false_positives += 1
                        ri += 1
                elif gc == separator:         #result doesn't have break here..
                    self.false_negatives += 1 #but gold does
                    gi += 1
                ri += 1
                gi += 1
        return (self.true_positives, self.false_positives, self.false_negatives)
                        
    def _calculate_precision(self, true_positives, false_positives):
        try:
            self.precision = true_positives / (true_positives + false_positives)
        except ZeroDivisionError:
            self.precision = 0
        return self.precision

    def _calculate_recall(self, true_positives, false_negatives):
        try:
            self.recall =  true_positives / (true_positives + false_negatives)
        except ZeroDivisionError:
            self.recall = 0
        return self.recall

    def _calculate_fscore(self, precision, recall, b=2):
        #https://en.wikipedia.org/wiki/F1_score
        try:
            self.fscore = (1+b**2) * (precision * recall) / ((b**2 * precision) + recall)
        except ZeroDivisionError:
            self.fscore = 0
        return self.fscore

class AlterParser(argparse.ArgumentParser):
    """Override behaviour of ArgumentParser.error() to print help message."""
    def error(self, msg):
        sys.stderr.write('error: %s\n\n' % msg)
        self.print_help()
        sys.exit(2)

def get_gold_segments(gold, separator='+'):
    """Read the gold file which contains correctly segmented words."""
    golds = []
    with open(gold, 'r', encoding='utf-8') as infile:
        for line in infile:
            gold = line.strip().split(separator)
            golds.append(gold)
    return golds

def test(segmenter, testing, gold, ver, write=True):
    print('Testing with') 
    print('  prefix parent threshold {}'.format(segmenter.ppth))
    print('  suffix parent threshold {}'.format(segmenter.spth))
    print('  prefix child threshold {}'.format(segmenter.pcth))
    print('  suffix child threshold {}'.format(segmenter.scth))
    evaluator = Evaluator()
    results = []
    with open(testing, 'r', encoding='utf-8') as infile:
        for line in infile:
            results.append(segmenter.segment(line))
    
    if write:
        output = 'results_{}.txt'.format(ver)
        with open(output, 'w', encoding='utf-8') as outfile: 
            for result in results:
                outfile.write('+'.join(result) + '\n')
    
    golds = get_gold_segments(gold)
    evaluator.calculate(results, golds)
    print('Morpheme breaks: results {}, gold {}'.format(
        evaluator.true_positives + evaluator.false_positives, 
        evaluator.true_positives + evaluator.false_negatives))
    print('Precision', evaluator.precision) 
    print('Recall', evaluator.recall)
    print('Fscore', evaluator.fscore)
    return evaluator
    
def sweep(training, testing, gold, ver, thresh, limit=3000):
    best_fscore = 0 
    ppth = 1.0
    spth = 1.0
    pcth = 1.0
    scth = 1.0
    best_ppth = ppth 
    best_spth = spth
    best_pcth = pcth
    best_scth = scth
    for i in range(1,10):
        i = i/10 
        ver = '{}_{}_{}'.format(ver, thresh, i)
        if thresh == 'ppth':
            ppth = i
        if thresh == 'spth':
            spth = i
        if thresh == 'pcth':
            pcth = i
        if thresh == 'scth':
            scth = i
        segmenter = Segmenter()
        for c in training:
            with open(c, 'r', encoding='utf-8') as infile:
                segmenter.add_corpus(infile, limit=limit)
        segmenter.train(ppth, spth, 
            pcth, scth, verbose=False)
        evaluator = test(segmenter, testing, gold, ver, write=False)
        if best_fscore < evaluator.fscore:
            best_fscore = evaluator.fscore
            best_ppth = ppth 
            best_spth = spth
            best_pcth = pcth
            best_scth = scth
    return (best_ppth, best_spth, best_pcth, best_scth, best_fscore)
    
def sweep_thresholds(training, testing, gold, ver):
    """
    Find a prefix and suffix threshold that maximize the f-score.
    """
    fp = 'segmodel_{}.pkl'.format(ver)
    try:   
        with open(fp, 'rb') as pkl:
            segmenter = pickle.load(pkl)
    except FileNotFoundError:
        print('Searching for thresholds that maximize performance...')
        ppth, x, y, z, best_fscore = sweep(training, testing, gold, ver, 'ppth')
        x, spth, y, z, best_fscore = sweep(training, testing, gold, ver, 'spth')
        x, y, pcth, z, best_fscore = sweep(training, testing, gold, ver, 'pcth')
        x, y, z, scth, best_fscore = sweep(training, testing, gold, ver, 'scth')
        print('Found best thresholds:')
        print('  prefix parent threshold {}'.format(ppth))
        print('  suffix parent threshold {}'.format(spth))
        print('  prefix child threshold {}'.format(pcth))
        print('  suffix child threshold {}'.format(scth))
        segmenter = Segmenter()
        for c in training:
            with open(c, 'r', encoding='utf-8') as infile:
                segmenter.add_corpus(infile)
        segmenter.train(ppth, spth, 
            pcth, scth, verbose=True)
        with open(fp, 'wb') as pkl:
            segmenter.save(pkl)
    else:
        print('Loaded previously trained model from {}'.format(fp))
        print('  which was trained on {} words, and thresholds:'.format(len(segmenter.corpus)))
        print('  prefix parent threshold {}'.format(segmenter.ppth))
        print('  suffix parent threshold {}'.format(segmenter.spth))
        print('  prefix child threshold {}'.format(segmenter.pcth))
        print('  suffix child threshold {}'.format(segmenter.scth))
    test(segmenter, testing, gold, ver)

def run_datasets():
    sys.setrecursionlimit(8000) #ja tree too deep for pkl due to long sentences
    print('Recursion limit set to:', sys.getrecursionlimit())

    ru_ver = 'ru_3.0.0'
    ru_train1 = 'data/ru_syntagrus_train.txt'
    ru_train2 = 'data/ru_lopatin_train.txt'
    ru_test = 'data/ru_dev.txt'
    ru_gold = 'data/ru_gold.txt'

    en_ver = 'en_3.0.0'
    en_train3 = 'data/en_words_train.txt'
    en_train1 = 'data/en_conll2000_train.txt'
    en_train2 = 'data/en_cmudict_train.txt'
    en_test = 'data/en_dev.txt'
    en_gold = 'data/en_gold.txt'

    ja_kanji_ver = 'ja_kanji_3.0.0'
    ja_kanji_train = 'data/ja_kanji_train.txt'
    ja_kanji_test = 'data/ja_kanji_dev.txt'
    ja_kanji_gold = 'data/ja_kanji_gold.txt'

    ja_kunrei_ver = 'ja_kunrei_3.0.0'
    ja_kunrei_train = 'data/ja_kunrei_train.txt'
    ja_kunrei_test = 'data/ja_kunrei_dev.txt'
    ja_kunrei_gold = 'data/ja_kunrei_gold.txt'

    print('\nWorking on Russian version {} ...'.format(ru_ver))
    sweep_thresholds([ru_train1, ru_train2], ru_test, ru_gold, ru_ver)
    print('\nWorking on English version {} ...'.format(en_ver))
    sweep_thresholds([en_train1, en_train2, en_train3], en_test, en_gold, en_ver)
    print('\nWorking on Japanese (kanji) version {} ...'.format(ja_kanji_ver))
    sweep_thresholds([ja_kanji_train], ja_kanji_test, ja_kanji_gold, ja_kanji_ver)
    print('\nWorking on Japanese (kunrei) version {} ...'.format(ja_kunrei_ver))
    sweep_thresholds([ja_kunrei_train], ja_kunrei_test, ja_kunrei_gold, ja_kunrei_ver)

def parse_args():
    """Parse command line arguments."""
    parser = AlterParser(prog='final.py', 
                         description='Use the RePortS algorithm to find morpheme breaks in words.')
    subparsers = parser.add_subparsers(dest='command') 
       
    parser_train = subparsers.add_parser('train',
                        help='train a new model (search for the best thresholds)')
    parser_test = subparsers.add_parser('test',
                        help='the version of the model to load')
    parser_standard = subparsers.add_parser('standard',
                        help='the version of the model to load')
 
    parser_test.add_argument('ver', 
                        help='the version of the model to load')
    parser_test.add_argument('test', 
                        help='one filepath to test corpus')
    parser_test.add_argument('gold', 
                        help='one filepath to gold corpus')

    parser_train.add_argument('ver', 
                        help='the version to save the model to')
    parser_train.add_argument('test', 
                        help='one filepath to test corpus (to use for finding thresholds)')
    parser_train.add_argument('gold', 
                        help='one filepath to gold corpus (to use for finding thresholds)')
    parser_train.add_argument('train', 
                        help='one or more corpus files to use', nargs='*')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.command == 'train':
        sweep_thresholds(args.train, args.test, args.gold, args.ver)
    elif args.command == 'test':
        fs = 'segmodel_{}.pkl'.format(args.ver)
        with open(fs, 'rb') as pkl:
            segmenter = pickle.load(pkl)
        test(segmenter, args.test, args.gold, args.ver)
    elif args.command == 'standard':
        run_datasets()
