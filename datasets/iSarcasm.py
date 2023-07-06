# The huggingface datasets loading scripts for iSarcasm Dataset

from typing import Any, Dict, List, Tuple
from huggingface_hub import HfApi
from datasets import Dataset, DatasetBuilder, DatasetInfo, Split, SplitGenerator, utils, DownloadManager, GeneratorBasedBuilder
import datasets
import csv

DATA_FILES = {
    "ar": {
        "train": "/Users/liyucheng/projects/cltk/data/iSarcasmEval-main/train/train.Ar.csv",
        "test": {
            'A': ["/Users/liyucheng/projects/cltk/data/iSarcasmEval-main/test/task_A_Ar_test.csv",],
            'C': ["/Users/liyucheng/projects/cltk/data/iSarcasmEval-main/test/task_C_Ar_test.csv",],
        },
    },
    "en": {
        "train": "/Users/liyucheng/projects/cltk/data/iSarcasmEval-main/train/train.En.csv",
        "test": {
            'A': ["/Users/liyucheng/projects/cltk/data/iSarcasmEval-main/test/task_A_En_test.csv",],
            'B': ["/Users/liyucheng/projects/cltk/data/iSarcasmEval-main/test/task_B_En_test.csv",],
            'C': ["/Users/liyucheng/projects/cltk/data/iSarcasmEval-main/test/task_C_En_test.csv",],
        },
    },
}

# iSarcasm_colums = ['tweet',
#  'sarcastic',
#  'rephrase',
#  'sarcasm',
#  'irony',
#  'satire',
#  'understatement',
#  'overstatement',
#  'rhetorical_question']

class iSarcasm(GeneratorBasedBuilder):
    VERSION = utils.Version("1.0.0")
    BUILDER_CONFIG_CLASS = datasets.BuilderConfig  # This will be a class you define for your configurations.
    BUILDER_CONFIGS = [
        BUILDER_CONFIG_CLASS(name="en", description="The english version of iSarcasm dataset"),
        BUILDER_CONFIG_CLASS(name="ar", description="The arabic version of iSarcasm dataset"),
    ]
    
    def _info(self) -> DatasetInfo:
        if self.config.name == "en":
            features = {"tweet": datasets.Value("string"), "sarcastic": datasets.Value("bool"), "rephrase": datasets.Value("string"), "sarcasm": datasets.Value("bool"), "irony": datasets.Value("bool"), "satire": datasets.Value("bool"), "understatement": datasets.Value("bool"), "overstatement": datasets.Value("bool"), "rhetorical_question": datasets.Value("bool"), "task": datasets.Value("string")}
        elif self.config.name == "ar":
            features = {"tweet": datasets.Value("string"), "sarcastic": datasets.Value("bool"), "dialect": datasets.Value("string"), "rephrase": datasets.Value("string"), "sarcasm": datasets.Value("bool"), "irony": datasets.Value("bool"), "satire": datasets.Value("bool"), "understatement": datasets.Value("bool"), "overstatement": datasets.Value("bool"), "rhetorical_question": datasets.Value("bool"), "task": datasets.Value("string")}
        return DatasetInfo(features=datasets.Features(features))
        
    def _split_generators(self, dl_manager: DownloadManager) -> List[SplitGenerator]:
        if self.config.name == "en":
            train_file_path = dl_manager.download_and_extract(DATA_FILES[self.config.name]["train"])
            test_file_paths = [(task, path) for task in DATA_FILES[self.config.name]["test"] for path in DATA_FILES[self.config.name]["test"][task]]
            return [SplitGenerator(name=Split.TRAIN, gen_kwargs={"filepath": train_file_path}),
                    SplitGenerator(name=Split.TEST, gen_kwargs={"filepath": test_file_paths})]
        elif self.config.name == "ar":
            train_file_path = dl_manager.download_and_extract(DATA_FILES[self.config.name]["train"])
            test_file_paths = [(task, path) for task in DATA_FILES[self.config.name]["test"] for path in DATA_FILES[self.config.name]["test"][task]]
            return [SplitGenerator(name=Split.TRAIN, gen_kwargs={"filepath": train_file_path}),
                    SplitGenerator(name=Split.TEST, gen_kwargs={"filepath": test_file_paths})]

    def _generate_examples(self, filepath: str) -> Tuple[Any, Dict]:
        if isinstance(filepath, list):
            for task, path in filepath:
                with open(path, encoding="utf-8") as f:
                    for id_, row in enumerate(f):
                        if id_ == 0:
                            continue
                        if task == 'A' and self.config.name == 'ar':
                            text,dialect,sarcastic = row.split(",")
                            yield id_, {"tweet": text.strip(), "sarcastic": bool(sarcastic.strip()), "dialect": dialect.strip(), "rephrase": '-', "sarcasm": '-', "irony": '-', "satire": '-', "understatement": '-', "overstatement": '-', "rhetorical_question": '-', "task": task}
                        if task == 'C' and self.config.name == 'ar':
                            text_0,text_1,dialect,sarcastic_id = row.split(",")
                            if sarcastic_id.strip() == '1':
                                yield id_, {"tweet": text_1.strip(), "sarcastic": bool(sarcastic_id.strip()), "dialect": dialect.strip(), "rephrase": text_0.strip(), "sarcasm": '-', "irony": '-', "satire": '-', "understatement": '-', "overstatement": '-', "rhetorical_question": '-', "task": task}
                            elif sarcastic_id.strip() == '0':
                                yield id_, {"tweet": text_0.strip(), "sarcastic": bool(sarcastic_id.strip()), "dialect": dialect.strip(), "rephrase": text_1.strip(), "sarcasm": '-', "irony": '-', "satire": '-', "understatement": '-', "overstatement": '-', "rhetorical_question": '-', "task": task}
                        
                        if task == 'A' and self.config.name == 'en':
                            text,sarcastic = row.split(",")
                            yield id_, {"tweet": text.strip(), "sarcastic": bool(sarcastic.strip()), "rephrase": '-', "sarcasm": '-', "irony": '-', "satire": '-', "understatement": '-', "overstatement": '-', "rhetorical_question": '-', "task": task}
                        if task == 'B' and self.config.name == 'en':
                            text,sarcasm,irony,satire,understatement,overstatement,rhetorical_question = row.split(",")
                            yield id_, {"tweet": text.strip(), "sarcastic": '-', "rephrase": '-', "sarcasm": bool(sarcasm.strip()), "irony": bool(irony.strip()), "satire": bool(satire.strip()), "understatement": bool(understatement.strip()), "overstatement": bool(overstatement.strip()), "rhetorical_question": bool(rhetorical_question.strip()), "task": task}
                        if task == 'C' and self.config.name == 'en':
                            text_0,text_1,sarcastic_id = row.split(",")
                            if sarcastic_id.strip() == '1':
                                yield id_, {"tweet": text_1.strip(), "sarcastic": bool(sarcastic_id.strip()), "rephrase": text_0.strip(), "sarcasm": '-', "irony": '-', "satire": '-', "understatement": '-', "overstatement": '-', "rhetorical_question": '-', "task": task}
                            elif sarcastic_id.strip() == '0':
                                yield id_, {"tweet": text_0.strip(), "sarcastic": bool(sarcastic_id.strip()), "rephrase": text_1.strip(), "sarcasm": '-', "irony": '-', "satire": '-', "understatement": '-', "overstatement": '-', "rhetorical_question": '-', "task": task}

        elif isinstance(filepath, str):
            with open(filepath, encoding="utf-8") as f:
                for id__, row in enumerate(f):
                    if id__ == 0:
                        continue
                    if self.config.name == 'ar':
                        id_,text,sarcastic,rephrase,dialect = list(csv.reader([row]))[0]
                        yield id_, {"tweet": text.strip(), "sarcastic": bool(sarcastic.strip()), "dialect": dialect.strip(), "rephrase": rephrase.strip(), "sarcasm": '-', "irony": '-', "satire": '-', "understatement": '-', "overstatement": '-', "rhetorical_question": '-', "task": '-'}
                    elif self.config.name == 'en':
                        try:
                            id_,tweet,sarcastic,rephrase,sarcasm,irony,satire,understatement,overstatement,rhetorical_question, = list(csv.reader([row]))[0]
                        except:
                            print('\n---', list(csv.reader([row])), '---')
                        yield id_, {"tweet": tweet.strip(), "sarcastic": bool(sarcastic.strip()), "rephrase": rephrase.strip(), "sarcasm": bool(sarcasm.strip()), "irony": bool(irony.strip()), "satire": bool(satire.strip()), "understatement": bool(understatement.strip()), "overstatement": bool(overstatement.strip()), "rhetorical_question": bool(rhetorical_question.strip()), "task": '-'}
