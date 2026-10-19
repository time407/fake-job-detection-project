from pathlib import Path
import json
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from create_report import add_bullet, add_page_number, add_table, format_run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
REPORT = ROOT / "Fake_Job_Detection_Report.docx"


def pct(v): return f"{100*v:.1f}%"


def main():
    payload = json.loads((OUT / "metrics.json").read_text())
    meta, models = payload["metadata"], payload["models"]
    final = models["fused"]
    advanced = json.loads((OUT / "advanced_metrics.json").read_text())
    chosen = advanced["advanced_ensemble_test"]
    baseline = advanced["linear_baseline_test"]
    errors = pd.read_csv(OUT / "advanced_test_errors.csv")
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Inches(8.5); sec.page_height = Inches(11)
    sec.top_margin = Inches(.72); sec.bottom_margin = Inches(.68)
    sec.left_margin = Inches(.78); sec.right_margin = Inches(.78)
    normal = doc.styles["Normal"]
    normal.font.name = "Aptos"; normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Aptos")
    normal.font.size = Pt(10.5); normal.font.color.rgb = RGBColor(30,30,30)
    normal.paragraph_format.space_after = Pt(6); normal.paragraph_format.line_spacing = 1.15
    for name, size, before, after in [("Title",26,0,14),("Heading 1",17,16,8),("Heading 2",12.5,11,5)]:
        st=doc.styles[name]; st.font.name="Aptos"; st.font.size=Pt(size); st.font.bold=True
        st.font.color.rgb=RGBColor(0,0,0); st.paragraph_format.space_before=Pt(before)
        st.paragraph_format.space_after=Pt(after); st.paragraph_format.keep_with_next=True
    title_ppr = doc.styles["Title"]._element.get_or_add_pPr()
    title_border = title_ppr.find(qn("w:pBdr"))
    if title_border is not None:
        title_ppr.remove(title_border)
    for s in doc.sections: add_page_number(s.footer.paragraphs[0])

    p=doc.add_paragraph(); p.paragraph_format.space_before=Pt(54)
    format_run(p.add_run("MACHINE LEARNING PROJECT REPORT"),11,True,"183B5B",font="Aptos")
    doc.add_paragraph("Fake Job Posting Detection",style="Title")
    p=doc.add_paragraph("A leakage-aware model combining job text, structured attributes, missingness, and formatting signals")
    format_run(p.runs[0],14,color="333333",font="Aptos")
    doc.add_picture(str(OUT/"figures"/"precision_recall.png"),width=Inches(6.3))
    p=doc.add_paragraph("Dataset  17,880 postings\nFraud prevalence  4.84%\nFinal model  XGBoost and high-dimensional text two-tower ensemble\nReport date  5 October 2026")
    p.paragraph_format.space_before=Pt(16)
    for r in p.runs: format_run(r,10.5,color="444444",font="Aptos")

    doc.add_page_break(); doc.add_heading("Executive Summary",1)
    p=doc.add_paragraph(); p.add_run("Use the advanced two-tower ensemble as a first-stage screening tool. ").bold=True
    p.add_run(f"On a grouped, untouched test set it achieved ROC AUC {chosen['roc_auc']:.3f} and PR AUC {chosen['pr_auc']:.3f}. "
              f"At the validation-selected threshold of {advanced['threshold']:.3f}, recall was {pct(chosen['recall'])} and precision was {pct(chosen['precision'])}. "
              "Among 3,450 test postings, the model found 165 fraudulent postings, missed 22, and produced 30 false alerts.")
    add_bullet(doc,"The model uses title, company profile, description, requirements, and benefits text while preserving section identity.")
    add_bullet(doc,"Character features capture spelling, URL, formatting, and phrase variants. Structured features cover categories, location, salary patterns, binary flags, missingness, and text shape.")
    add_bullet(doc,"XGBoost learns nonlinear interactions in the structured tower; a separate sparse text tower preserves high-dimensional word and character evidence. Validation-selected blending combines their complementary strengths.")
    add_bullet(doc,"Duplicate full-text templates stay in one split, reducing leakage between training and evaluation data.")
    add_bullet(doc,"Scores should prioritize human review, not trigger automatic rejection. The operating threshold should reflect review capacity and the cost of missed fraud.")
    add_table(doc,["Metric","Result","Interpretation"],[
        ["PR AUC",f"{chosen['pr_auc']:.3f}","Primary ranking metric for the rare fraud class; random baseline is about 0.048"],
        ["ROC AUC",f"{chosen['roc_auc']:.3f}","Probability that a random fraud case outranks a random legitimate case"],
        ["Recall",pct(chosen['recall']),"165 of 187 fraud cases detected"],
        ["Precision",pct(chosen['precision']),"About two thirds of alerts were confirmed fraud"],
        ["Balanced accuracy",pct(chosen['balanced_accuracy']),"Average sensitivity across both classes"],
    ],widths=[1.25,1.25,4.1])

    doc.add_heading("Data and Signal Design",1)
    doc.add_paragraph(f"The file contains {meta['rows']:,} postings and {meta['positive_rows']} fraud labels, a prevalence of {pct(meta['positive_rate'])}. "
                      "A classifier that always predicts legitimate would appear about 95% accurate, so raw accuracy is not a useful primary metric.")
    doc.add_picture(str(OUT/"figures"/"missingness.png"),width=Inches(6.25)); doc.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER
    p=doc.add_paragraph("Figure 1  Missingness by field"); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph("Salary range is missing in 84.0% of rows and department in 64.6%. Missing values are retained as explicit categories or indicators because completeness can carry behavioral information. job_id is retained only for traceability and never used as a predictor.")
    add_table(doc,["Feature family","Encoding","Information retained"],[
        ["Five text sections","Word one- and two-gram TF-IDF","Vocabulary and phrases"],
        ["Five text sections","Character three- to five-gram TF-IDF","Spelling, formatting, URLs, and word variants"],
        ["Categorical fields","One-hot encoding with rare-level control","Industry, function, experience, education, and employment type"],
        ["Location","Country, region, and city components","Multiple geographic levels"],
        ["Salary","Lower bound, upper bound, width, and missing flag","Numeric content and completion behavior"],
        ["Text shape","Length, word count, uppercase, digits, URLs, email, punctuation","Writing and formatting patterns"],
        ["Binary fields","Direct input","Remote work, company logo, and screening questions"],
    ],widths=[1.25,2.35,3.0])

    doc.add_heading("Leakage-Aware Evaluation",1)
    doc.add_paragraph(f"Normalized full text produced {meta['unique_text_groups']:,} unique groups, so 2,131 rows share complete text with at least one other row. "
                      "Every group is assigned wholly to training, validation, or testing. The validation split chooses the threshold; the test split is reserved for final reporting.")
    add_table(doc,["Split","Rows","Fraud rate","Purpose"],[
        ["Training",f"{meta['split_sizes']['train']:,}",pct(meta['split_positive_rates']['train']),"Fit vectorizers and classifier"],
        ["Validation",f"{meta['split_sizes']['validation']:,}",pct(meta['split_positive_rates']['validation']),"Select the F2-optimal threshold"],
        ["Test",f"{meta['split_sizes']['test']:,}",pct(meta['split_positive_rates']['test']),"One-time final evaluation"],
    ],widths=[1.2,1.2,1.2,3.0])
    doc.add_paragraph("F2 gives recall more weight than precision. This matches a screening setting where a missed fraudulent posting is assumed to cost more than an additional manual review. If operational costs differ, the threshold should change without retraining the full representation.")
    doc.add_heading("Model Architecture",1)
    doc.add_paragraph("The final system uses two specialized towers. A word-and-character TF-IDF tower models 200,000 high-dimensional text patterns. An XGBoost tower learns nonlinear interactions among 1,268 categorical, missingness, salary, location, binary, and text-shape features. Validation selected weights of 0.48 for the text tower and 0.52 for XGBoost, followed by a separate F2-oriented decision threshold.")
    doc.add_paragraph("This architecture captures interactions such as missing profile information conditional on industry, location, and text length while retaining detailed phrase evidence. Feature associations remain correlational; no location, industry, or word should be treated as fraud evidence on its own.")

    doc.add_heading("Results and Operating Point",1)
    doc.add_picture(str(OUT/"figures"/"model_comparison.png"),width=Inches(6.15)); doc.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER
    p=doc.add_paragraph("Figure 2  Model comparison on the grouped holdout set"); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    labels={"structured":"Structured only","word_text":"Word text only","fused":"Fused model"}; rows=[]
    for key in ("structured","word_text","fused"):
        m=models[key]["test_selected"]; rows.append([labels[key],f"{m['pr_auc']:.3f}",f"{m['roc_auc']:.3f}",pct(m['precision']),pct(m['recall']),f"{m['f2']:.3f}"])
    rows.append(["Advanced ensemble",f"{chosen['pr_auc']:.3f}",f"{chosen['roc_auc']:.3f}",pct(chosen['precision']),pct(chosen['recall']),f"{chosen['f2']:.3f}"])
    add_table(doc,["Model","PR AUC","ROC AUC","Precision","Recall","F2"],rows,widths=[1.55,.9,.9,.9,.9,.8])
    doc.add_paragraph(f"The advanced ensemble improves PR AUC from {baseline['pr_auc']:.3f} to {chosen['pr_auc']:.3f}, ROC AUC from {baseline['roc_auc']:.3f} to {chosen['roc_auc']:.3f}, and F2 from {baseline['f2']:.3f} to {chosen['f2']:.3f}. Precision rises from {pct(baseline['precision'])} to {pct(chosen['precision'])}; recall falls from {pct(baseline['recall'])} to {pct(chosen['recall'])}, making the advanced operating point more suitable when review capacity is limited.")
    add_table(doc,["Threshold","Precision","Recall","False alerts","Missed fraud","Use case"],[
        ["0.498 advanced",pct(chosen['precision']),pct(chosen['recall']),"30","22","Higher alert precision"],
        ["0.299 linear",pct(baseline['precision']),pct(baseline['recall']),"98","11","Prioritize low miss rate"],
    ],widths=[1.4,.9,.9,.8,.8,1.8])

    doc.add_heading("Error Analysis",1)
    doc.add_paragraph("At the selected advanced threshold, 22 fraud cases were missed and 30 legitimate postings were flagged. Missed cases often have neutral titles or fewer suspicious text patterns. False alerts may resemble risky templates, locations, industries, or formatting found in the training data. These examples support targeted review and do not imply that a title or place is inherently suspicious.")
    fn=errors[errors.error_type=="false_negative"].sort_values("score",ascending=False).head(5)
    fp=errors[errors.error_type=="false_positive"].sort_values("score",ascending=False).head(5)
    add_table(doc,["Type","job_id","Title","Score"],
              [["Missed",int(r.job_id),str(r.title)[:46],f"{r.score:.3f}"] for _,r in fn.iterrows()]+
              [["False alert",int(r.job_id),str(r.title)[:46],f"{r.score:.3f}"] for _,r in fp.iterrows()],widths=[1.0,.8,4.0,.9])

    doc.add_heading("Deployment and Monitoring",1)
    add_bullet(doc,"Validate the incoming schema, encoding, missingness, and text fields; preserve job_id for traceability.")
    add_bullet(doc,"Generate a 0-to-1 risk score and route postings above the approved threshold to human review.")
    add_bullet(doc,"Store reviewer decisions and appeals as new labels for recalibration and retraining.")
    add_bullet(doc,"Monitor score distribution, alert rate, field missingness, and reviewer confirmation rate every week.")
    add_bullet(doc,"Compare false-alert rates across region, industry, and employment type to detect unfair concentration.")
    doc.add_paragraph("The project includes reproducible training and prediction scripts, fixed split indices, full metrics, error records, feature coefficients, figures, and a serialized model. predict.py adds fraud_risk_score and fraud_prediction to a compatible CSV.")

    doc.add_heading("Limitations and Next Steps",1)
    add_bullet(doc,"Labels may contain errors and may not represent current fraud tactics.")
    add_bullet(doc,"All splits come from one source; cross-platform, temporal, and multilingual generalization remain untested.")
    add_bullet(doc,"Scores are useful for ranking but have not been calibrated as real-world fraud probabilities.")
    add_bullet(doc,"Exact text grouping does not remove near-duplicate paraphrases. Semantic clustering would create a stricter future test.")
    doc.add_paragraph("The next evaluation should use newly collected, time-ordered data. Compare the current linear model with calibrated gradient boosting and a pretrained language model, while preserving grouped or temporal holdouts and measuring review cost and fairness.")

    doc.add_heading("Reproducibility",1)
    add_table(doc,["Artifact","Purpose"],[
        ["code/train.py","Feature engineering, grouped splitting, training, evaluation, and export"],
        ["code/predict.py","Batch scoring for new CSV files"],
        ["code/train_advanced.py","XGBoost tower training and validation-weighted model fusion"],
        ["code/predict_advanced.py","Batch scoring with the final two-tower ensemble"],
        ["outputs/metrics.json","Validation and test metrics"],
        ["outputs/split_indices.json","Fixed train, validation, and test indices"],
        ["outputs/test_errors.csv","False alerts and missed fraud cases"],
        ["outputs/feature_importance.csv","Largest positive and negative model coefficients"],
        ["outputs/fake_job_detector.joblib","Transformers, classifier, and selected threshold"],
        ["outputs/fake_job_detector_advanced.joblib","Final XGBoost and sparse-text ensemble"],
    ],widths=[2.3,4.4])
    doc.add_paragraph("Reproduce with: python code/train.py --data data/fake_job_postings.csv --output outputs, then PYTHONPATH=code python code/train_advanced.py. The random seed is 42 and dependencies are recorded in requirements.txt.")
    doc.core_properties.title="Fake Job Posting Detection Machine Learning Report"
    doc.core_properties.subject="Model design, leakage-aware evaluation, and deployment guidance"
    doc.core_properties.author=""
    doc.save(REPORT); print(REPORT)


if __name__=="__main__": main()
