# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval. Copy this file and complete it with the info for your task.

This file generally create just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.

Author:
"""
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


GERMAN_MMLU_SUBSETS = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

class CustomGermanMMLUTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function="mmlu_prompt",
            hf_repo="zeitgeist-ai/mmlu_de",
            metric=["loglikelihood_acc_norm"],
            hf_avail_splits=["test", "validation"],
            evaluation_splits=["test"],
            few_shots_split="validation",
            few_shots_select="sequential",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            output_regex=None,
            frozen=False,
            trust_dataset=True,
            version=0,
        )


GERMAN_MMLU_TASKS = [
    CustomGermanMMLUTask(name=f"mmlu_de:{subset}", hf_subset=subset) for subset in GERMAN_MMLU_SUBSETS
] 


# DEFINE YOUR PROMPT FUNCTIONS
# Define as many as you need for your different tasks
def mmlu_prompt(line, task_name: str = None):
    """MMLU prompt without letters"""
    topic = line["subject"]
    prompt = f"Im Folgenden werden Fragen gestellt zu {topic.replace('_', ' ')}.\nFrage: "
    prompt += line["question"] + "\nAntwort:"

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[f" {c}" for c in line["choices"]],
        gold_index=line["answer"],
        instruction=f"Im Folgenden werden Fragen gestellt zu {topic.replace('_', ' ')}.\n",
    )


# STORE YOUR EVALS
_TASKS = GERMAN_MMLU_TASKS

# MODULE LOGIC
# You should not need to touch this
# Convert to dict for lighteval
TASKS_TABLE = [task.as_dict() for task in _TASKS]

if __name__ == "__main__":
    print(t["name"] for t in TASKS_TABLE)
    print(len(TASKS_TABLE))
