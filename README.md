# cs224nMediBert

MEDI-BERPT: A Novel Multitask Approach
to Streamlining Chinese Healthcare

In the pursuit of revolutionizing the medical industry, Large Language Mod-
els (LLMs) have faced obstacles due to stringent quality requirements and
numerous ethical concerns. Previous medical NLP approaches focused on
building single-task models that were dedicated to simple classification or
to generating doctor-like responses. Our study presents a multi-task trans-
former encoder-decoder model that takes a patient’s symptom descriptions
and generates predictions for both the appropriate medical department and
an initial diagnosis or a follow-up question. The primary objective is to
evaluate the extent to which the inclusion of a secondary task, specifically
the doctor-diagnosis generation, enhances the performance of the medical
department classification task.
We leverage a dataset comprising 3 million doctor-patient conversations
scraped from Chinese online medical forums (MedDialog) to fine-tune
BERT and GPT-2 models, establishing baseline performances. Our findings
demonstrate that incorporating a secondary objective enables the model to
capture more nuanced relationships between the doctor’s response and the
patient’s symptom description, consequently enhancing the medical depart-
ment classification process.
The proposed multi-objective transformer encoder-decoder model outper-
forms the original BERT encoder with an accuracy of 91% compared to
84% in predicting the top 10 most relevant medical departments. Nonethe-
less, our analysis highlights significant limitations in attempting to gener-
ate a doctor’s response based solely on the patient’s symptom description,
underscoring the importance of developing AI-assisted tools that support
patients within the medical system, rather than seeking to replace human
expertise.


## To install all libraries in requirements.txt, run
pip install -r requirements.txt
