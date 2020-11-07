#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""WikiReading Evaluator."""

import argparse
import sys
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Set, TextIO, Tuple, Union

import pandas as pd


def parse_args():
    """Parse CLI arguments.

    Returns:
        namespace: namespace with parsed variables.

    """
    parser = argparse.ArgumentParser('Information Extractor Evaluator')
    parser.add_argument(
        '--prediction',
        '-p',
        type=argparse.FileType('r', encoding='utf-8'),
        required=True,
        help="Path to file with model's predictions.")
    parser.add_argument(
        '--reference',
        '-r',
        type=argparse.FileType('r', encoding='utf-8'),
        required=True,
        help='Path to the reference file.')
    parser.add_argument(
        '-d',
        dest='separator',
        type=str,
        default='=',
        help='Property-value delimiter')
    parser.add_argument(
        '--metric',
        default='mean-F1',
        choices=['F1', 'mean-F1'],
        help='Metric to compute (default: mean-F1).')
    parser.add_argument(
        '--properties',
        type=argparse.FileType('r', encoding='utf-8'),
        help='Property set to be limitted to')
    parser.add_argument(
        '--ignore-case', '-i',
        action='store_true', default=False,
        help='Property set to be limitted to')
    parser.add_argument(
        '--output-file', '-o',
        default=sys.stdout,
        type=argparse.FileType('w', encoding='utf-8'),
        help='Save the results to the file (default: print to stdout)')
    return parser.parse_args()


class FScorer:
    """Corpus level F1 Score evaluator."""

    def __init__(self):
        """Initialize class."""
        self.__precision = []
        self.__recall = []

    def add(self, out_items: List[str], ref_items: List[str]):
        """Add more items for computing corpus level scores.

        Args:
            out_items: outs from a single document (line)
            ref_items: reference of the evaluated document (line)

        """
        ref_items_copy = ref_items.copy()
        indicators = []
        for pred in out_items:
            if pred in ref_items_copy:
                indicators.append(1)
                ref_items_copy.remove(pred)
            else:
                indicators.append(0)
        self.__add_to_precision(indicators)

        indicators = []
        out_items_copy = out_items.copy()
        for ref in ref_items:
            if ref in out_items_copy:
                indicators.append(1)
                out_items_copy.remove(ref)
            else:
                indicators.append(0)
        self.__add_to_recall(indicators)

    def __add_to_precision(self, item: List[int]):
        if isinstance(item, list):
            self.__precision.extend(item)
        else:
            self.__precision.append(item)

    def __add_to_recall(self, item: List[int]):
        if isinstance(item, list):
            self.__recall.extend(item)
        else:
            self.__recall.append(item)

    def precision(self) -> float:
        """Compute precision.

        Returns:
            float: corpus level precision

        """
        if self.__precision:
            precision = sum(self.__precision) / len(self.__precision)
        else:
            precision = 0.0
        return precision

    def recall(self) -> float:
        """Compute recall.

        Returns:
            float: corpus level recall

        """
        if self.__recall:
            recall = sum(self.__recall) / len(self.__recall)
        else:
            recall = 0.0
        return recall

    def f_score(self) -> float:
        """Compute F1 score.

        Returns:
            float: corpus level F1 score.

        """
        precision = self.precision()
        recall = self.recall()
        if precision or recall:
            fscore = 2 * precision * recall / (precision + recall)
        else:
            fscore = 0.0
        return fscore

    def false_negative(self) -> int:
        """Return the number of false negatives.

        Returns:
            int: number of false negatives.

        """
        return len(self.__recall) - sum(self.__recall)

    def false_positive(self) -> int:
        """Return the number of false positives.

        Returns:
            int: number of false positives.

        """
        return len(self.__precision) - sum(self.__precision)

    def true_positive(self) -> int:
        """Return number of true positives.

        Returns:
            int: number of true positives.

        """
        return sum(self.__precision)

    def condition_positive(self) -> int:
        """Return number of condition positives.

        Returns:
            int: number of condition positives.

        """
        return len(self.__precision)


def property_scores_to_string(
        general_fscorer: FScorer,
        property_fscorers: Dict[str, FScorer],
        print_format: str = 'text'
) -> str:
    """Print out scores per property.

    Args:
        general_fscorer: General FScorer
        property_fscorers: a dict with property fscorers.
        print_format: output format: text or latex

    Returns:
        str: string table with feature scores.

    """
    final_results = [(property_name,
                      property_fscorers[property_name].precision(),
                      property_fscorers[property_name].recall(),
                      property_fscorers[property_name].f_score())
                     for property_name in sorted(property_fscorers.keys())]
    final_results.append(('ALL',
                          general_fscorer.precision(),
                          general_fscorer.recall(),
                          general_fscorer.f_score()))
    df = pd.DataFrame(final_results, columns=['Label', 'Precision', 'Recall', 'F-1'])
    df.set_index('Label', drop=True, inplace=True)
    out: str
    if print_format == 'latex':
        out = df.reset_index().to_latex(index=False)
    elif print_format == 'text':
        out = df.reset_index().to_string(index=False)
    elif print_format == 'json':
        out = df.to_json(orient='index')

    return out


class ArnoldEvaluator:
    """Arnold Evaluator."""

    def __init__(
        self,
        reference: List[List[str]],
        answers: List[List[str]],
        separator: str = '=',
        property_set: Optional[Set[str]] = None,
        ignore_case: bool = False,
    ):
        """Initialize ArnoldEvaluator.

        Arguments:
            reference: reference
            answers: answers to be evaluated
            separator: property name and property value separator
            property_set: if given, the score will be computed taking into account only these properties.
            ignore_case: if true, compute scores ignoring casing.

        """
        self.reference = reference
        self.answers = answers
        self.separator = separator
        self.property_set = property_set
        self.ignore_case = ignore_case
        self.__general_scorer, self.__property_scorers = self._evalute_f1()

    @property
    def general_scorer(self) -> FScorer:
        """Get general scorer.

        Returns:
            FScorer: the general scorer.

        """
        return self.__general_scorer

    @property
    def property_scorers(self) -> DefaultDict[str, FScorer]:
        """Get a scorer for each property.

        Returns:
            Fscorer: the general scorer.

        """
        return self.__property_scorers

    def filter_properties(self, items: List[str], values: Union[str, List[str], Set[str]]) -> List[str]:
        """Filter the list of properties by provided property name(s).

        Args:
            items: a list with properties
            values: a property name(s)

        Returns:
            list: a filtered properties

        """
        if isinstance(values, str):
            values = [values]
        return list(filter(lambda x: x.split(self.separator)[0] in values, items))

    def _evalute_f1(self) -> Tuple[FScorer, DefaultDict[str, FScorer]]:
        """Evaluate the output file.

        Returns:
            tuple: generatl fscorer and a dict with fscorer per label.

        """
        label_fscorers: DefaultDict[str, FScorer] = defaultdict(FScorer)
        general_fscorer = FScorer()
        reference_labels: Set[str] = set()
        for ans_items, ref_items in zip(self.answers, self.reference):
            if self.ignore_case:
                ans_items = self.uppercase_items(ans_items)
                ref_items = self.uppercase_items(ref_items)

            if self.property_set:
                ans_items = self.filter_properties(ans_items, self.property_set)
                ref_items = self.filter_properties(ref_items, self.property_set)

            reference_labels |= set(item.split(self.separator)[0] for item in ref_items)
            for label in set(item.split(self.separator)[0] for item in ref_items + ans_items):
                if self.property_set and label not in self.property_set:
                    continue
                if label == '':
                    continue
                label_out = self.filter_properties(ans_items, label)
                label_ref = self.filter_properties(ref_items, label)
                label_fscorers[label].add(label_out, label_ref)

            if len(ans_items) == 1 and ans_items[0] == '' and len(ref_items) == 1 and ref_items[0] == '':
                continue

            general_fscorer.add(ans_items, ref_items)

        for label in list(label_fscorers.keys()):
            if label not in reference_labels:
                del label_fscorers[label]

        return general_fscorer, label_fscorers

    def uppercase_items(self, items: List[str]) -> List[str]:
        """Upperecase item values.

        Args:
            items: items which values should ba uppercased.

        Returns:
            list: with with uppercased values.

        """
        uppercased = []
        for item in items:
            try:
                name, value = item.split(self.separator, maxsplit=1)
                uppercased.append(f'{name}{self.separator}{value.upper()}')
            except ValueError:
                print(f'Item in wrong format: "{item}".', file=sys.stderr)
                #  uppercased.append(item)

        return uppercased

    def get_score(self, metric: str) -> float:
        """Get F1 score depenging on the version.

        Args:
            metric: F1 version: F1 or F1-mean

        Returns:
            float: score

        """
        if metric == 'mean-F1':
            scores: List[float] = []
            for ans_items, ref_items in zip(self.answers, self.reference):
                if self.property_set:
                    ans_items = self.filter_properties(ans_items, self.property_set)
                    ref_items = self.filter_properties(ref_items, self.property_set)
                ref_items = list(filter(bool, ref_items))
                if len(ref_items) == 0 and len(ans_items) == 0:
                    continue
                fscorer = FScorer()
                fscorer.add(ans_items, ref_items)
                scores.append(fscorer.f_score())
            return sum(scores) / len(scores)
        elif metric == 'F1':
            return self.general_scorer.f_score()
        return 0.0

    def line_by_line(self):
        """Compute scores line by line.

        Returns:
            List: list with scorers.

        """
        scores = []
        for ans_items, ref_items in zip(self.answers, self.reference):
            line_scores = []
            fscorer = FScorer()
            fscorer.add(ans_items, ref_items)
            line_scores.append({
                'property': 'ALL',
                'F-1': fscorer.f_score(),
                'reference-len': len(ref_items),
            })

            for label in set(item.split(self.separator, maxsplit=1)[0] for item in ref_items):
                label_out = [item for item in ans_items if item.split(self.separator, maxsplit=1)[0] == label]
                label_ref = [item for item in ref_items if item.split(self.separator, maxsplit=1)[0] == label]

                fscorer = FScorer()
                fscorer.add(label_out, label_ref)
                line_scores.append({
                    'property': label,
                    'F-1': fscorer.f_score(),
                    'reference-len': len(label_ref),
                })
            scores.append(line_scores)
        return scores


def cli_main(args: argparse.Namespace):
    """CLI main.

    Args:
        args: cli arguments
    """
    reference = [line.strip().split(' ') for line in args.reference]
    answers = [line.strip().split(' ') for line in args.out_file]

    separator = args.separator

    property_set: Optional[Set[str]]
    if args.properties:
        property_set = args.properties.read().strip().split(' ')
    else:
        property_set = None

    arnold = ArnoldEvaluator(
        reference,
        answers,
        separator,
        property_set,
        args.ignore_case,
    )

    args.save(arnold.get_score(args.return_score))

def evaluate(
    prediction_file: TextIO,
    reference_file: TextIO,
    separator: str,
    output_file: TextIO,
    metric: str,
    properties: Optional[TextIO] = None,
    ignore_case: bool = False
):
    reference = [line.strip().split(' ') for line in reference_file]
    answers = [line.strip().split(' ') for line in prediction_file]

    property_set: Optional[Set[str]]
    if properties:
        property_set = set(properties.read().strip().split(' '))
    else:
        property_set = None

    arnold = ArnoldEvaluator(
        reference,
        answers,
        separator,
        property_set,
        ignore_case,
    )
    score = arnold.get_score(metric)
    output_file.write(f'{score}\n')


def main() -> None:
    """Main."""
    args = parse_args()
    evaluate(
        prediction_file=args.prediction,
        reference_file=args.reference,
        separator=args.separator,
        output_file=args.output_file,
        metric=args.metric,
        properties=args.properties,
        ignore_case=args.ignore_case,
    )


if __name__ == '__main__':
    main()

