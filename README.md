### AI Hiring Bias Audit
A fairness-driven machine learning project that analyzes and mitigates potential bias in hiring decisions based on resume data.

### Dataset Description
The dataset simulates job applicant profiles with the following fields:

=
| **Feature**                | **Description**                                                   |
|----------------------------|-------------------------------------------------------------------|
| Age, ExperienceYears,      | Numerical hiring factors   
|  SkillScore, InterviewScore |                                                                   |
| Gender                     | Sensitive attribute: 0 = Male, 1 = Female                         |
| RecruitmentStrategy        | 1 = Aggressive, 2 = Moderate, 3 = Conservative                    |
| HiringDecision             | Target variable: 1 = Hired, 0 = Not Hired                         |
| text                       | Synthesized resume text derived from structured data              |


### Model Architecture

1. TF-IDF + Logistic Regression
Input: Text-only.

Vectorizer: Unigrams + bigrams.

Classifier: Logistic Regression.

Bias audit: Gender words analyzed via SHAP.

Counterfactual & fairness analysis.

### Fairness Audit
No strong gender disparity observed. Gendered words had negligible SHAP attribution, meaning the model relied on skill- and merit-based indicators.

## Explainability
Used shap.Explainer() on both text and hybrid models. 

Explained 5 cases (3 hire, 2 no-hire). 

Top drivers: SkillScore, InterviewScore, ExperienceYears. Negligible influence of gender indicators. 

## Bias Mitigation
Counterfactual Data Augmentation.

## Key Learnings
Recruitment strategy influences hiring outcomes more than gender.

SHAP showed model is merit-based, not biased.

Gender imbalance in training alone did not cause unfair outcomes.


### TO-DO
Refactor into bigger project for automated end-to-end resume screening. 

Deploy bias dashboard (e.g., with Streamlit or Gradio).

Extend to class-based bias (socio-economic status of the applicant's address) and other potential biases. 
