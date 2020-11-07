# From Dataset Recycling to Multi-Property Extraction and Beyond

This repository contains model implementations and data from the paper: _From Dataset Recycling to Multi-Property Extraction and Beyond_ by Tomasz Dwojak, Tomasz, Michał Pietruszka, Łukasz Borchmann, Jakub Chłędowski, and Filip Graliński.

[[BibTeX](./ref.bib)]

## Contents
 1. [WikiReading Recycled](#wikireading-recycled)
 2. [Models](#models)
 3. [Reproduction of the results](#reproduction-of-the-results)
 
## WikiReading Recycled

The _WikiReading Recycled_ and _WikiReading_ are based on the same data (articles from Wikipedia combined with Wikidata), yet differ in how they are arranged.
The dataset is available at: [https://applica-public.s3-eu-west-1.amazonaws.com/multi-property-extraction/wikireading-recycled.tar](https://applica-public.s3-eu-west-1.amazonaws.com/multi-property-extraction/wikireading-recycled.tar).

### Multi-Property Extraction
Instances from the original _WikiReading_ dataset were merged to produce over 4M samples in the Multi-Property Extraction (MPE) paradigm. In the MPE, the system is expected to return values for multiple properties at once. Hence, it can be considered a generalization of a single property extraction task as it can be easily formulated as the MPE. 

### Human-annotated test set
The quality of test sets plays a pivotal role in reasoning about a system's performance. Therefore, a group of annotators went through the test set instances and assessed whether the value either appeared in the article or can be inferred from it. To make further analysis possible, we provide both datasets, before (test-A) and after (test-B) annotation.

### Diagnostic subsets
Moreover, we determined auxiliary validation subsets with specific qualities to help improve data analysis and provide additional information at different stages of developing a system. Please refer to the paper for a detailed description.

Characteristics of different systems can be compared qualitatively by evaluating on these subsets. For instance, the _long articles_ subset is challenging for systems that consume truncated inputs. _Unseen_ is precisely constructed to assess systems' ability to extract previously not seen properties. On the other hand, _rare_ can be viewed as an approximation of the system's performance on a lower-resource downstream extraction task.

### Dataset Construction
Instead of performing a random split, we carefully divide the data assuming that 20% of properties should appear solely in the test set (more precisely, not seen before in train and validation sets). Around one thousand articles containing properties not seen in the remaining subsets were drafted to achieve the mentioned objective. Similarly, properties unique for the validation set were introduced to enable approximation of the test set performance without disclosing particular labels.

Additionally, test and validation sets share 10% of the properties that do not appear in the train set, increasing these subsets' size by 2,000 articles each. Another 2,000 articles containing the same properties as the train set were added to each validation and test set. All the remaining articles were used to produce the training set.

To sum up, we achieved a design where as much as 50% of the properties cannot be seen in the training split, while the remaining 50% of the properties can appear in any split. We chose these properties carefully so that the test and validation sets' size does not exceed 5,000 articles.

## Models
We made public the following models on WikiReading Recycled:
 * [Vanilla Dual-Source Transformer](https://applica-public.s3-eu-west-1.amazonaws.com/multi-property-extraction/fairseq-models/dual-source-transformer.tar.gz)
 * [Dual-Source RoBERTa](https://applica-public.s3-eu-west-1.amazonaws.com/multi-property-extraction/fairseq-models/dual-source-roberta.tar.gz)
 * [T5](https://applica-public.s3-eu-west-1.amazonaws.com/multi-property-extraction/fairseq-models/t5.tar.gz)

The Fairseq implementation of these models is available in the [fairseq_modules](./tutorial/fairseq_modules) directory.

## Reproduction of the Results
We prepared a [short tutorial](./tutorial/README.md) on how to reproduce the results from the paper.

