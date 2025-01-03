# @paper Redescription mining-based business process deviance analysis
# @author Engjëll Ahmeti & https://towardsdatascience.com/how-to-evaluate-text-generation-models-metrics-for-automatic-evaluation-of-nlp-models-e1c251b04ec1
# @date 3/3/2022

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
import itertools
import pyter
import re

class Metrics:
    #region BLEU = Bilingual Evaluation Understudy Score
    def bleu(self, ref, gen):
        '''
        calculate pair wise bleu score. uses nltk implementation
        Args:
            references : a list of reference sentences
            candidates : a list of candidate(generated) sentences
        Returns:
            bleu score(float)
        '''
        ref_bleu = []
        gen_bleu = []
        for l in gen:
            gen_bleu.append(l.split())
        for i,l in enumerate(ref):
            ref_bleu.append([l.split()])
        cc = SmoothingFunction()
        try:
            score_bleu = corpus_bleu(ref_bleu, gen_bleu, weights=(0, 1, 0, 0), smoothing_function=cc.method4)
        except Exception as e:
            if hasattr(e, 'message'):
                score_bleu = e.message 
            else:
                score_bleu = e
           
        return score_bleu
    #endregion

    #region ROUGE = Recall Oriented Understudy for Gisting Evaluation
    def _split_into_words(self, sentences):
        """Splits multiple sentences into words and flattens the result"""
        return list(itertools.chain(*[_.split(" ") for _ in sentences]))

    def _get_word_ngrams(self, n, sentences):
        """Calculates word n-grams for multiple sentences.
        """
        assert len(sentences) > 0
        assert n > 0

        words = self._split_into_words(sentences)
        return self._get_ngrams(n, words)

    def _get_ngrams(self, n, text):
        """Calcualtes n-grams.
            Args:
            n: which n-grams to calculate
            text: An array of tokens
            Returns:
            A set of n-grams
        """
        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i:i + n]))
        return ngram_set

    def rouge_n(self, reference_sentences, evaluated_sentences, n=2):
        """
        Computes ROUGE-N of two text collections of sentences.
        Source: http://research.microsoft.com/en-us/um/people/cyl/download/
        papers/rouge-working-note-v1.3.1.pdf
        Args:
        evaluated_sentences: The sentences that have been picked by the summarizer
        reference_sentences: The sentences from the referene set
        n: Size of ngram.  Defaults to 2.
        Returns:
        recall rouge score(float)
        Raises:
        ValueError: raises exception if a param has len <= 0
        """
        if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
            raise ValueError("Collections must contain at least 1 sentence.")

        evaluated_ngrams = self._get_word_ngrams(n, evaluated_sentences)
        reference_ngrams = self._get_word_ngrams(n, reference_sentences)
        reference_count = len(reference_ngrams)
        evaluated_count = len(evaluated_ngrams)

        # Gets the overlapping ngrams between evaluated and reference
        overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
        overlapping_count = len(overlapping_ngrams)

        # Handle edge case. This isn't mathematically correct, but it's good enough
        if evaluated_count == 0:
            precision = 0.0
        else:
            precision = overlapping_count / evaluated_count

        if reference_count == 0:
            recall = 0.0
        else:
            recall = overlapping_count / reference_count

        f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

        #just returning recall count in rouge, useful for our purpose
        return recall
    #endregion

    #region TER = Translation Edit Rate
    def ter(self, ref, gen):
        '''
        Args:
            ref - reference sentences - in a list
            gen - generated sentences - in a list
        Returns:
            averaged TER score over all sentence pairs
        '''
        if len(ref) == 1:
            total_score = pyter.ter(gen[0].split(), ref[0].split())
        else:
            total_score = 0
            for i in range(len(gen)):
                try:
                    total_score = total_score + pyter.ter(gen[i].split(), ref[i].split())
                except:
                    total_score = total_score + 1.0
            total_score = total_score/len(gen)
        return total_score
    #endregion
