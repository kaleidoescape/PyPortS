from math import log2
from collections import OrderedDict
import unittest
import pyports
import json

class Tree(unittest.TestCase):

    def test_forward_tree(self):
        tree = pyports.Tree()
        tree.increment(list('car'), False)
        tree.increment(list('cart'), False)
        tree.increment(list('cot'), False)
        tree.log_normalize()
        answer = {"count": 3, "prob": log2(1)-log2(3), 'chain_prob': log2(1)-log2(3), "subtrees": 
                    {"c": 
                        {"count": 3, "prob": 0.0, 'chain_prob': log2(1)-log2(3), "subtrees": 
                            {"a": 
                               {"count": 2, "prob": log2(2)-log2(3), 'chain_prob': (log2(1)-log2(3))+(log2(2)-log2(3)), "subtrees": 
                                   {"r": 
                                       {"count": 2, "prob": 0.0, 'chain_prob': (log2(1)-log2(3))+(log2(2)-log2(3)), "subtrees": 
                                           {"t": 
                                               {"count": 1, "prob": log2(1)-log2(2), 'chain_prob': (log2(1)-log2(3))+(log2(2)-log2(3))+(log2(1)-log2(2)), "subtrees": 
                                                   {}
                                               }
                                           }
                                       }
                                   }
                               },
                            "o": 
                                {"count": 1, "prob": log2(1)-log2(3), 'chain_prob': (log2(1)-log2(3))+(log2(1)-log2(3)), "subtrees": 
                                    {"t": 
                                        {"count": 1, "prob": 0.0, 'chain_prob': (log2(1)-log2(3))+(log2(1)-log2(3)), "subtrees":
                                            {} 
                                        }
                                    }
                                } 
                            }
                        }
                    }
                 }
        result = tree._to_dict() 
        assert result == answer, 'Correct: {}\n\nWas: {}'.format(answer, result)
    
    def test_backward_tree(self):
        tree = pyports.Tree()
        tree.increment(list('car'), True)
        tree.increment(list('cart'), True)
        tree.increment(list('cot'), True)
        tree.log_normalize()
        answer = {"count": 3, "prob": log2(1)-log2(3), 'chain_prob': log2(1)-log2(3), "subtrees": 
                    {"t": 
                        {"count": 2, "prob": log2(2)-log2(3), 'chain_prob': (log2(1)-log2(3))+(log2(2)-log2(3)), "subtrees": 
                            {"o": 
                                {"count": 1, "prob": log2(1)-log2(2), 'chain_prob': (log2(1)-log2(3))+(log2(2)-log2(3))+(log2(1)-log2(2)), "subtrees": 
                                    {"c": 
                                        {"count": 1, "prob": 0.0, 'chain_prob': (log2(1)-log2(3))+(log2(2)-log2(3))+(log2(1)-log2(2)), "subtrees": 
                                            {}
                                        }
                                    }
                                }, 
                            "r": 
                                {"count": 1, "prob": log2(1)-log2(2), 'chain_prob': (log2(1)-log2(3))+(log2(2)-log2(3))+(log2(1)-log2(2)), "subtrees": 
                                    {"a": 
                                        {"count": 1, "prob": 0.0, 'chain_prob': (log2(1)-log2(3))+(log2(2)-log2(3))+(log2(1)-log2(2)), "subtrees": 
                                            {"c": 
                                                {"count": 1, "prob": 0.0, 'chain_prob': (log2(1)-log2(3))+(log2(2)-log2(3))+(log2(1)-log2(2)), "subtrees":
                                                    {}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }, 
                    "r": 
                        {"count": 1, "prob": log2(1)-log2(3), 'chain_prob': (log2(1)-log2(3))+(log2(1)-log2(3)), "subtrees": 
                            {"a": 
                                {"count": 1, "prob": 0.0, 'chain_prob': (log2(1)-log2(3))+(log2(1)-log2(3)), "subtrees": 
                                    {"c": 
                                        {"count": 1, "prob": 0.0, 'chain_prob': (log2(1)-log2(3))+(log2(1)-log2(3)), "subtrees": 
                                            {}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
        result = tree._to_dict()  
        assert result == answer, 'Correct: {}\n\nWas: {}'.format(answer, result)

    def test_from_corpus(self):
        corpus = ['car', 'cart', 'cot']
        tree = pyports.Tree()
        tree.from_corpus(corpus)
        answer = {"count": 3, "prob": log2(1)-log2(3), 'chain_prob': log2(1)-log2(3), "subtrees": 
                    {"c": 
                        {"count": 3, "prob": 0.0, 'chain_prob': log2(1)-log2(3), "subtrees": 
                            {"a": 
                               {"count": 2, "prob": log2(2)-log2(3), 'chain_prob': (log2(1)-log2(3))+(log2(2)-log2(3)), "subtrees": 
                                   {"r": 
                                       {"count": 2, "prob": 0.0, 'chain_prob': (log2(1)-log2(3))+(log2(2)-log2(3)), "subtrees": 
                                           {"t": 
                                               {"count": 1, "prob": log2(1)-log2(2), 'chain_prob': (log2(1)-log2(3))+(log2(2)-log2(3))+(log2(1)-log2(2)), "subtrees": 
                                                    {}
                                               }
                                           }
                                       }
                                   }
                               },
                            "o": 
                                {"count": 1, "prob": log2(1)-log2(3), 'chain_prob': (log2(1)-log2(3))+(log2(1)-log2(3)), "subtrees": 
                                    {"t": 
                                        {"count": 1, "prob": 0.0, 'chain_prob': (log2(1)-log2(3))+(log2(1)-log2(3)), "subtrees": 
                                            {}
                                        }
                                    }
                                } 
                            }
                        }
                    }
                 }
        result = tree._to_dict() 
        assert result == answer, 'Correct: {}\n\nWas: {}'.format(answer, result)
       
class TestSegmenter(unittest.TestCase):
    def test_prefix_scores(self):
        corpus = ['preseared', 'preheated']
        correct_prefixes = {'pre': 38}
        segmenter = pyports.Segmenter()
        segmenter.add_corpus(corpus)
        segmenter.train()
        answer_prefixes = segmenter._to_dict(segmenter.prefix_scores)
        assert answer_prefixes == correct_prefixes, 'Correct: {}\nWas: {}'.format(correct_prefixes, answer_prefixes)

    def test_suffix_scores(self):
        corpus = ['preseared', 'preheated']
        correct_suffixes = {'ed': 38} 
        segmenter = pyports.Segmenter()
        segmenter.add_corpus(corpus)
        segmenter.train()
        answer_suffixes = segmenter._to_dict(segmenter.suffix_scores)
        assert answer_suffixes == correct_suffixes, 'Correct: {}\nWas: {}'.format(correct_suffixes, answer_suffixes)

    def test_prefix_probs(self):
        corpus = ['preseared', 'preheated']
        correct_prefixes = {'pre': log2(1)-log2(2)}
        segmenter = pyports.Segmenter()
        segmenter.add_corpus(corpus)
        segmenter.train()
        answer_prefixes = segmenter._to_dict(segmenter.prefix_probs)
        assert answer_prefixes == correct_prefixes, 'Correct: {}\nWas: {}'.format(correct_prefixes, answer_prefixes)

    def test_suffix_probs(self):
        corpus = ['preseared', 'preheated']
        correct_suffixes = {'ed': log2(1)-log2(2)} 
        segmenter = pyports.Segmenter()
        segmenter.add_corpus(corpus)
        segmenter.train()
        answer_suffixes = segmenter._to_dict(segmenter.suffix_probs)
        assert answer_suffixes == correct_suffixes, 'Correct: {}\nWas: {}'.format(correct_suffixes, answer_suffixes)

    def test_find_suffix(self):
        corpus = ['preseared', 'preheated']
        correct_suffixes = ['ed'] 
        segmenter = pyports.Segmenter()
        segmenter.add_corpus(corpus)
        segmenter.train()
        word, answer_suffixes = segmenter._find_suffixes('prefeared') 
        assert answer_suffixes == correct_suffixes, 'Correct: {}\nWas: {}'.format(correct_suffixes, answer_suffixes)

    def test_find_suffixes(self):
        corpus = ['dry', 'dryness', 'drying', 
                  'polite', 'politeness', 'politenesses', 
                  'fox', 'foxes', 
                  'hash', 'hashing', 'hashes']
        segmenter = pyports.Segmenter()
        segmenter.add_corpus(corpus)
        segmenter.train()
        word, answer_suffixes = segmenter._find_suffixes('politenesses') 
        correct_suffixes = ['ness', 'es'] 
        assert answer_suffixes == correct_suffixes, 'Correct: {}\nWas: {}'.format(correct_suffixes, answer_suffixes)

    def test_find_prefixes(self):
        corpus = ['dry', 'predry', 'undry',
                  'heat', 'preheat', 'repreheat', 'unheat',
                  'turn', 'return', 'repaint', 'retry'] 
        segmenter = pyports.Segmenter()
        segmenter.add_corpus(corpus)
        segmenter.train()
        word, answer_prefixes = segmenter._find_prefixes('repreheat') 
        correct_prefixes = ['re', 'pre']
        assert answer_prefixes == correct_prefixes, 'Correct: {}\nWas: {}'.format(correct_prefixes, answer_prefixes)

class TestEvaluator(unittest.TestCase):
    def test_count(self):
        dev = [['pre', 'cogni', 'tion'], ['devalue'], ['evalua', 'te'], ['ef', 'fect']]
        gold = [['pre', 'cogni', 'tion'], ['de', 'value'], ['eval', 'uate'], ['effect']]
        correct_tp = 2
        correct_fp = 2
        correct_fn = 2
        evaluator = pyports.Evaluator()
        tp, fp, fn = evaluator._count(dev, gold)
        assert tp == correct_tp, 'Correct: {}\nWas: {}'.format(correct_tp, tp)
        assert fp == correct_fp, 'Correct: {}\nWas: {}'.format(correct_tp, fp)
        assert fn == correct_fn, 'Correct: {}\nWas: {}'.format(correct_tp, fn)

if __name__ == '__main__':
    unittest.main()

