# Reproducing the Results

We provide most of the models mentioned in the final results (Table 6 and 7 in the paper).
We used [fairseq](https://github.com/pytorch/fairseq) to implement and train the models.
You can install the dependencies by executing:

```
pip install -r requirements.txt
```

Additionally, we used the `luigi` library for all the staff related to preprocessing and postprocessing the data.
Finally, we provide an evaluation script to determine the scores.

## Dual-Source Models
Dual-Source models (vanilla dual-source transformer and dual-source RoBERTa) share the same preprocessing process but use different vocabularies.
We train a new [sentencepiece](https://github.com/google/sentencepiece) model on all training data for dual-source transformer and reuse the original RoBERTa vocabulary for Dual-source RoBERTa.

The input to these models consists of two parts: the article and the requested properties. The properties are separated by the `###` token. For instance:
```
William Costello Kennedy, PC (August 27, 1868 -- January 17, 1923) was a Canadian politician. Born in Ottawa, Ontario, he was first elected to the Canadian House of Commons in the riding of Essex North in the 1917 federal election as a Laurier-Liberal. He was re-elected as a Liberal in 1921. From 1921 until his death, he was the Minister of Railways and Canals in the government of William Lyon Mackenzie King.
family name ### position held ### date of birth ### member of political party ### instance of ### date of death ### occupation ### sex or gender ### given name ### country of citizenship
```
The first line is a Wikipedia article, and the second contains the requested properties.

## Evaluation of Models
We published three models on WikiReading Recycled:
 * T5,
 * vanilla dual-source transformer,
 * dual-source RoBERTa.

Moreover, we prepared a complete pipeline to reproduce our results. To evaluate a model, run:
 ```
 PYTHONPATH=. luigi --local-scheduler --module tutorial_scripts EvaluateModelTask --model MODEL --split SPLIT
 ```
 where `MODEL` may be one of `T5`, `DUAL_SOURCE_TRANSFORMER`, `DUAL_ROBERTA_TRANSFORMER`; and split one of `dev-0`, `test-A`, `test-B`.

 For example:
 ```
 PYTHONPATH=. luigi --local-scheduler --module tutorial_scripts EvaluateModelTask --model DUAL_ROBERTA_TRANSFORMER --split test-B
 ```
 will evaluate dual-source RoBERTa model on test-B.

You may specify a diagnostic subset by adding `--subset SUBSET` option (`SUBSET`: _unseen_, _rare_, _categorical_, _relational_, _exact-match_, _long-articles_).
 For example:
 ```
 PYTHONPATH=. luigi --local-scheduler --module tutorial_scripts EvaluateModelTask --model DUAL_ROBERTA_TRANSFORMER --split test-B --subset rare
 ```

 The results are written to `results/DUAL_ROBERTA_TRANSFORMER/test-B.rare`.

 The schema of directories is as follows:
 * _dataset_: the WikiReading Recycled dataset directory
 * _models_: the models
 * _processed_: the preprocessed data 
 * _binarized_: the binary version of preprocessed files (required by fairseq)
 * _outputs_: model's raw predictions and they post-processed version
 * _results_: the final results
 * _fairseq_modules_: implementation of the models in fairseq
 * _tutorial_scripts_: the pipeline implementation

In case of any problem, contact [Tomasz Dwojak](mailto:tomasz.dwojak@applica.ai) or report an issue.
