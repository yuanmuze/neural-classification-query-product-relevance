from __future__ import annotations
import json, os, random, re, time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import joblib, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

LABELS = ["E", "S", "C", "I"]
TEXT = ["query", "product_title"]

def norm_text(x):
    return re.sub(r"\s+", " ", x.strip()) if isinstance(x, str) else x

def word_count(s): return len(str(s).split())
def lfs_guard(path: Path):
    with path.open("rb") as f: head = f.read(120)
    if head.startswith(b"version https://git-lfs.github.com/spec"):
        raise RuntimeError(f"{path} is a Git LFS pointer, not Parquet. Follow data/README.md to run git lfs pull.")

def write_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

def audit(df, name):
    out=[]
    for c in df.columns:
        missing=int(df[c].isna().sum()); blank=int(df[c].map(lambda x:isinstance(x,str) and not x.strip()).sum())
        out.append({"dataset":name,"field":c,"dtype":str(df[c].dtype),"rows":len(df),"missing_count":missing,"missing_pct":round(100*missing/max(len(df),1),4),"blank_text_count":blank})
    return pd.DataFrame(out)

def load_raw(data_dir: Path):
    ex=data_dir/"shopping_queries_dataset_examples.parquet"; pr=data_dir/"shopping_queries_dataset_products.parquet"
    for p in (ex,pr):
        if not p.exists(): raise FileNotFoundError(f"Missing {p}. See data/README.md; no synthetic fallback is used.")
        lfs_guard(p)
    ecols=["example_id","query","query_id","product_id","product_locale","esci_label","large_version","split"]
    # Descriptions/brand/colour are deliberately not model inputs.  Loading the
    # very large optional text fields would also be wasteful on a CPU laptop.
    pcols=["product_id","product_locale","product_title"]
    examples=pq.read_table(ex, columns=ecols, filters=[("large_version","=",1),("product_locale","=","us")]).to_pandas()
    products=pq.read_table(pr, columns=pcols, filters=[("product_locale","=","us")]).to_pandas()
    return examples, products

def clean_and_join(examples, products, out: Path):
    logs=[]
    def log(rule, n, action, reason, remaining): logs.append({"rule":rule,"affected_count":int(n),"action":action,"reason":reason,"remaining_rows":int(remaining)})
    before_e=audit(examples,"examples_before"); before_p=audit(products,"products_before")
    keydupes=products.duplicated(["product_id","product_locale"],keep=False)
    conflict_keys=products.loc[keydupes,["product_id","product_locale"]].drop_duplicates()
    if len(conflict_keys):
        conflict_keys.to_csv(out/"audit/product_key_conflicts.csv",index=False)
        products=products.merge(conflict_keys.assign(_bad=1),on=["product_id","product_locale"],how="left"); n=products._bad.notna().sum(); products=products[products._bad.isna()].drop(columns="_bad"); log("product_composite_key_conflict",n,"drop product keys and matching examples","many-to-one merge cannot be trusted for ambiguous product metadata",len(products))
        examples=examples.merge(conflict_keys.assign(_bad=1),on=["product_id","product_locale"],how="left"); n=examples._bad.notna().sum(); examples=examples[examples._bad.isna()].drop(columns="_bad"); log("examples_with_product_key_conflict",n,"drop","conservative isolation of ambiguous product keys",len(examples))
    merged=examples.merge(products,on=["product_id","product_locale"],how="left",validate="many_to_one",indicator=True)
    log("left_join_unmatched_products",(merged._merge!="both").sum(),"drop","title unavailable for query-title classifier",len(merged))
    merged=merged[merged._merge=="both"].drop(columns="_merge")
    for col in TEXT: merged[col]=merged[col].where(merged[col].notna(),None).map(norm_text)
    invalid=(~merged.esci_label.isin(LABELS)) | merged.esci_label.isna(); n=invalid.sum(); merged=merged[~invalid]; log("invalid_or_missing_label",n,"drop","target outside E/S/C/I cannot be modeled",len(merged))
    for col in TEXT:
        bad=merged[col].isna() | merged[col].eq(""); n=bad.sum(); merged=merged[~bad]; log(f"missing_or_blank_{col}",n,"drop","pair text required for current model",len(merged))
    full=merged.duplicated(keep="first"); n=full.sum(); merged=merged[~full]; log("fully_duplicate_records",n,"drop","identical duplicate",len(merged))
    pair=["query","product_id","product_locale"]
    conflicts=merged.groupby(pair).esci_label.nunique(); badpairs=conflicts[conflicts>1].reset_index()[pair]
    if len(badpairs): badpairs.to_csv(out/"audit/conflicting_label_pairs.csv",index=False); marked=merged.merge(badpairs.assign(_bad=1),on=pair,how="left"); n=marked._bad.notna().sum(); merged=marked[marked._bad.isna()].drop(columns="_bad"); log("conflicting_query_product_labels",n,"drop all conflicting pairs","no defensible automatic target resolution",len(merged))
    samepair=merged.duplicated(pair,keep="first"); n=samepair.sum(); merged=merged[~samepair]; log("same_label_duplicate_pairs",n,"drop","retain one repeated pair after conflicts removed",len(merged))
    merged["query_norm"]=merged["query"].map(norm_text).str.casefold(); merged["query_chars"]=merged["query"].str.len(); merged["title_chars"]=merged["product_title"].str.len(); merged["query_words"]=merged["query"].map(word_count); merged["title_words"]=merged["product_title"].map(word_count)
    for c in ["query_chars","title_chars"]:
        q1,q3=merged[c].quantile([.25,.75]); lo,hi=q1-1.5*(q3-q1),q3+1.5*(q3-q1); merged[c+"_length_outlier"]=(merged[c]<lo)|(merged[c]>hi)
    after=audit(merged,"joined_clean_after")
    pd.concat([before_e,before_p,after]).to_csv(out/"audit/data_audit.csv",index=False); pd.DataFrame(logs).to_csv(out/"audit/cleaning_log.csv",index=False)
    return merged

def choose_groups(df, target, seed):
    groups=df.query_norm.drop_duplicates().sample(frac=1,random_state=seed).tolist(); selected=[]; n=0
    sizes=df.groupby("query_norm").size()
    for g in groups:
        if n>=target: break
        selected.append(g); n+=sizes[g]
    return df[df.query_norm.isin(selected)].copy()

def splits(df, target_train, target_test, seed, out):
    dev=choose_groups(df[df.split=="train"],target_train,seed); test=choose_groups(df[df.split=="test"],target_test,seed+1)
    overlap=set(dev.query_norm)&set(test.query_norm); removed=dev.query_norm.isin(overlap).sum(); dev=dev[~dev.query_norm.isin(overlap)].copy()
    groups=dev.query_norm.to_numpy(); gss=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed); tr,va=next(gss.split(dev,groups=groups)); train,valid=dev.iloc[tr].copy(),dev.iloc[va].copy()
    for name,x in [("train",train),("validation",valid),("test",test)]: x[["example_id","query_norm"]].to_csv(out/"manifests"/f"{name}_manifest.csv",index=False)
    checks={"train_validation_normalized_query_overlap":len(set(train.query_norm)&set(valid.query_norm)),"dev_test_normalized_query_overlap_removed":int(removed),"official_train_test_query_id_overlap":len(set(df[df.split=='train'].query_id)&set(df[df.split=='test'].query_id)),"product_overlap_train_test":len(set(train.product_id)&set(test.product_id))}
    write_json(checks,out/"audit/split_leakage_checks.json"); return train,valid,test

def numeric(df):
    qtokens=df["query"].map(lambda s:set(s.casefold().split())); ptokens=df["product_title"].map(lambda s:set(s.casefold().split()))
    overlap=np.array([len(a&b) for a,b in zip(qtokens,ptokens)]); qn=np.array([max(1,len(x)) for x in qtokens]); pn=np.array([max(1,len(x)) for x in ptokens])
    return np.c_[df.query_words,df.title_words,overlap/np.array([max(1,len(a|b)) for a,b in zip(qtokens,ptokens)]),overlap/qn]

@dataclass
class FeatureBundle:
    vectorizer: object; svd: object; scaler: object; input_dim: int
    def transform(self, df):
        q=self.svd.transform(self.vectorizer.transform(df["query"])); p=self.svd.transform(self.vectorizer.transform(df["product_title"])); n=self.scaler.transform(numeric(df)); return np.hstack([q,p,np.abs(q-p),q*p,n]).astype("float32")

def fit_features(train, max_features, requested_svd):
    vec=TfidfVectorizer(ngram_range=(1,2),max_features=max_features,lowercase=True,dtype=np.float32)
    corpus=pd.concat([train["query"],train["product_title"]]).tolist(); mat=vec.fit_transform(corpus); components=max(2,min(requested_svd,mat.shape[0]-1,mat.shape[1]-1)); svd=TruncatedSVD(n_components=components,random_state=42).fit(mat); scaler=StandardScaler().fit(numeric(train)); q=svd.transform(vec.transform(train["query"])); dim=4*q.shape[1]+4
    return FeatureBundle(vec,svd,scaler,dim)

class MLP(nn.Module):
    def __init__(self,d): super().__init__(); self.net=nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,4))
    def forward(self,x): return self.net(x)

def metric(y,p): return {"accuracy":accuracy_score(y,p),"macro_f1":f1_score(y,p,average="macro",zero_division=0),"weighted_f1":f1_score(y,p,average="weighted",zero_division=0)}
def train_mlp(xtr,ytr,xv,yv,cfg,device):
    model=MLP(xtr.shape[1]).to(device); opt=torch.optim.Adam(model.parameters(),lr=.001); lossfn=nn.CrossEntropyLoss(); loader=DataLoader(TensorDataset(torch.tensor(xtr),torch.tensor(ytr)),batch_size=cfg["batch_size"],shuffle=True); best=(-1,None); rows=[]; wait=0
    for epoch in range(1,cfg["epochs"]+1):
        start=time.time(); model.train(); losses=[]
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad(); loss=lossfn(model(xb),yb); loss.backward(); opt.step(); losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            tr_logits=model(torch.tensor(xtr).to(device)); va_logits=model(torch.tensor(xv).to(device))
            tr_pred=tr_logits.argmax(1).cpu().numpy(); pred=va_logits.argmax(1).cpu().numpy()
            val_loss=float(lossfn(va_logits,torch.tensor(yv).to(device)).item())
        tr_m=metric(ytr,tr_pred); m=metric(yv,pred)
        rows.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"validation_loss":val_loss,"train_accuracy":tr_m["accuracy"],"train_macro_f1":tr_m["macro_f1"],"validation_accuracy":m["accuracy"],"validation_macro_f1":m["macro_f1"],"seconds":time.time()-start})
        if m["macro_f1"]>best[0]: best=(m["macro_f1"],{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}); wait=0
        else: wait+=1
        if wait>=cfg["patience"]: break
    model.load_state_dict(best[1]); return model,rows

def save_plots(train, out):
    plotdir=out/"figures"; plotdir.mkdir(exist_ok=True); sns.set_theme(style="whitegrid")
    plots=[("class_distribution",lambda: sns.countplot(data=train,x="esci_label",order=LABELS)),("query_word_distribution",lambda: sns.histplot(train.query_words,bins=30)),("title_word_distribution",lambda: sns.histplot(train.title_words,bins=30)),("title_length_by_class",lambda: sns.boxplot(data=train,x="esci_label",y="title_chars",order=LABELS)),("products_per_query",lambda: sns.histplot(train.groupby("query_norm").size(),bins=30))]
    for name,fn in plots:
        plt.figure(figsize=(8,4)); fn(); plt.title(name.replace("_"," ").title()); plt.tight_layout(); plt.savefig(plotdir/f"{name}.png",dpi=150); plt.close()
    miss=train.isna().mean().sort_values(ascending=False); plt.figure(figsize=(8,4)); sns.barplot(x=miss.index,y=miss.values); plt.xticks(rotation=45,ha="right"); plt.ylabel("Missing rate"); plt.title("Key Field Missing Rate"); plt.tight_layout(); plt.savefig(plotdir/"missing_rate.png",dpi=150); plt.close()

def run(config, profile):
    cfg=config[profile]; out=Path(config["output_dir"]); [p.mkdir(parents=True,exist_ok=True) for p in [out,out/"audit",out/"figures",out/"models",out/"predictions",out/"manifests"]]
    seed=config["seed"]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    failure = out/"audit"/"run_failure.log"
    if failure.exists(): failure.unlink()
    ex,pr=load_raw(Path(config["data_dir"])); raw_join_candidate_rows=len(ex); df=clean_and_join(ex,pr,out); train,valid,test=splits(df,cfg["train_target_rows"],cfg["test_target_rows"],seed,out); save_plots(train,out)
    plt.figure(figsize=(6,4)); sns.barplot(x=["Before cleaning","After cleaning"],y=[raw_join_candidate_rows,len(df)]); plt.ylabel("Rows"); plt.title("Rows Before and After Cleaning"); plt.tight_layout(); plt.savefig(out/"figures"/"cleaning_row_comparison.png",dpi=150); plt.close()
    eda=[]
    for n,x in [("train",train),("validation",valid),("test",test)]:
        for c in ["query_chars","title_chars","query_words","title_words"]: eda.append({"split":n,"feature":c,**x[c].describe().to_dict(),"variance":x[c].var()})
    pd.DataFrame(eda).to_csv(out/"audit/numeric_descriptives.csv",index=False)
    fb=fit_features(train,cfg["max_features"],cfg["svd_components"]); xtr,xv,xt=fb.transform(train),fb.transform(valid),fb.transform(test); mapping={x:i for i,x in enumerate(LABELS)}; ytr=train.esci_label.map(mapping).to_numpy(); yv=valid.esci_label.map(mapping).to_numpy(); yt=test.esci_label.map(mapping).to_numpy()
    joblib.dump(fb,out/"models/features.joblib"); write_json({"labels":LABELS,"input_dim":fb.input_dim,"profile":profile,"seed":seed},out/"models/model_metadata.json")
    models={}; dummy=DummyClassifier(strategy="most_frequent").fit(xtr,ytr); log=LogisticRegression(max_iter=500,n_jobs=-1,random_state=seed).fit(xtr,ytr); models["dummy"]=(dummy,dummy.predict(xt),None); models["logistic_regression"]=(log,log.predict(xt),log.predict_proba(xt)); joblib.dump(dummy,out/"models/dummy.joblib"); joblib.dump(log,out/"models/logistic_regression.joblib")
    device="cuda" if torch.cuda.is_available() else "cpu"; mlp,history=train_mlp(xtr,ytr,xv,yv,cfg,device); torch.save(mlp.state_dict(),out/"models/mlp.pt"); mlp.eval()
    with torch.no_grad(): probs=torch.softmax(mlp(torch.tensor(xt).to(device)),1).cpu().numpy()
    models["mlp"]=(mlp,probs.argmax(1),probs); pd.DataFrame(history).to_csv(out/"audit/mlp_history.csv",index=False)
    h=pd.DataFrame(history)
    plt.figure(figsize=(7,4)); plt.plot(h.epoch,h.train_loss,label="Train loss"); plt.plot(h.epoch,h.validation_loss,label="Validation loss"); plt.xlabel("Epoch"); plt.ylabel("Cross-entropy loss"); plt.legend(); plt.title("MLP Training and Validation Loss"); plt.tight_layout(); plt.savefig(out/"figures/mlp_loss_curves.png",dpi=150); plt.close()
    plt.figure(figsize=(7,4)); plt.plot(h.epoch,h.train_macro_f1,label="Train Macro-F1"); plt.plot(h.epoch,h.validation_macro_f1,label="Validation Macro-F1"); plt.plot(h.epoch,h.validation_accuracy,label="Validation Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Score"); plt.legend(); plt.title("MLP Training and Validation Metrics"); plt.tight_layout(); plt.savefig(out/"figures/mlp_validation_curves.png",dpi=150); plt.close()
    compare=[]
    for name,(_,pred,probs) in models.items():
        m=metric(yt,pred); compare.append({"model":name,**m}); report=classification_report(yt,pred,labels=range(4),target_names=LABELS,output_dict=True,zero_division=0); write_json(report,out/f"audit/{name}_classification_report.json"); cm=confusion_matrix(yt,pred,labels=range(4)); pd.DataFrame(cm,index=LABELS,columns=LABELS).to_csv(out/f"audit/{name}_confusion_counts.csv"); pd.DataFrame(cm/np.maximum(cm.sum(1,keepdims=True),1),index=LABELS,columns=LABELS).to_csv(out/f"audit/{name}_confusion_normalized.csv")
    pd.DataFrame(compare).to_csv(out/"audit/model_comparison.csv",index=False)
    cp=pd.DataFrame(compare).melt(id_vars="model",value_vars=["accuracy","macro_f1","weighted_f1"],var_name="metric",value_name="score")
    plt.figure(figsize=(8,4)); sns.barplot(data=cp,x="model",y="score",hue="metric"); plt.ylim(0,1); plt.xlabel("Model"); plt.ylabel("Test score"); plt.title("Test Metric Comparison"); plt.tight_layout(); plt.savefig(out/"figures/model_comparison.png",dpi=150); plt.close()
    cm=confusion_matrix(yt,models["mlp"][1],labels=range(4))
    plt.figure(figsize=(6,5)); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=LABELS,yticklabels=LABELS); plt.xlabel("Predicted label"); plt.ylabel("True label"); plt.title("MLP Test Confusion Matrix"); plt.tight_layout(); plt.savefig(out/"figures/mlp_test_confusion_matrix.png",dpi=150); plt.close()
    pred=models["mlp"][1]; probs=models["mlp"][2]; rows=test[["example_id","query","product_title","esci_label"]].copy(); rows["predicted_label"]=[LABELS[i] for i in pred]
    for i,l in enumerate(LABELS): rows[f"prob_{l}"]=probs[:,i]
    rows.to_csv(out/"predictions/mlp_test_predictions.csv",index=False)
    examples=pd.concat([rows[rows.esci_label==rows.predicted_label].head(5).assign(result="correct"),rows[rows.esci_label!=rows.predicted_label].head(10).assign(result="incorrect")]); examples.to_csv(out/"predictions/error_analysis_examples.csv",index=False)
    # saved-loaded MLP prediction verification
    restored=MLP(fb.input_dim); restored.load_state_dict(torch.load(out/"models/mlp.pt",map_location="cpu",weights_only=True)); restored.eval()
    with torch.no_grad(): restored_pred=restored(torch.tensor(xt[:min(100,len(xt))])).argmax(1).numpy()
    checks=json.loads((out/"audit/split_leakage_checks.json").read_text()); checks.update({"merge_preserved_after_cleaning":"validated many_to_one before unmatched removal","labels_valid":bool(df.esci_label.isin(LABELS).all()),"mlp_output_dimension":4,"preprocessing_fit_split":"train_only","saved_loaded_prediction_consistent":bool(np.array_equal(restored_pred,pred[:len(restored_pred)])),"device":device,"sizes":{"train":len(train),"validation":len(valid),"test":len(test)}}); write_json(checks,out/"audit/run_checks.json")
    summary={"profile":profile,"source_url":"https://github.com/amazon-science/esci-data","access_date":"2026-09-16","metrics":compare,"sizes":checks["sizes"]}
    write_json(summary,out/"run_summary.json")
    report=Path("report"); report.mkdir(exist_ok=True)
    score=pd.DataFrame(compare).round(4).to_markdown(index=False)
    if profile != "assignment":
        return checks
    (report/"report_draft.md").write_text(f"""# Neural Classification of Query-Product Relevance

## 1. Problem and data source
This coursework treats ESCI as four-class query-product relevance classification: Exact (E), Substitute (S), Complement (C), and Irrelevant (I). It does not claim to be a personalised recommender or a complete ranking system. Data source: {summary['source_url']} (accessed {summary['access_date']}). The official README identifies `large_version == 1` as the Task 2 version. The run used US-locale pairs joined on `(product_locale, product_id)`.

## 2. Data inspection and cleaning
The pipeline writes field-level before/after audit results to `outputs/audit/data_audit.csv` and every applied rule to `outputs/audit/cleaning_log.csv`. It removes invalid targets, absent/blank required pair text, unmatched products, exact duplicates, same-label duplicate pairs, and all ambiguous conflicting-label pairs. Product-key conflicts are isolated rather than silently selecting a record. Whitespace is normalised after missing values are handled. Length outliers are flagged using IQR fences and not automatically discarded.

## 3. Exploration
The formal-training split has {len(train):,} pairs, {train.query_norm.nunique():,} normalized queries and {train.product_id.nunique():,} products; validation has {len(valid):,} pairs and official-test-source evaluation has {len(test):,} pairs. `outputs/audit/numeric_descriptives.csv` contains count, mean, median (50%), variance, standard deviation, range and quartiles. The saved figures show class counts, missingness, query/title word distributions, title length by class, and products per query. Interpretations must be based on those generated artifacts rather than assumed patterns.

## 4. Development method and leakage controls
Official train and test were retained as separate sources. An approximately 80/20 `GroupShuffleSplit` within official train used normalized query text as the group. Any development query that also occurred in the selected official test source was removed before the train/validation split. The leakage and persistence checks are in `outputs/audit/run_checks.json`. TF-IDF, SVD and numeric scaling were fit only on final training rows. The feature representation is shared training-vocabulary unigram/bigram TF-IDF reduced by SVD, then `[q, p, |q-p|, q*p]` plus simple lengths and token-overlap features. It is lightweight and interpretable, but has limited semantic understanding.

## 5. Models and evaluation
The comparison uses the identical test manifest for most-frequent DummyClassifier, Logistic Regression, and a PyTorch MLP (input→128 ReLU Dropout 0.3→64 ReLU Dropout 0.2→4 logits). MLP selection used validation Macro-F1 and patience-based early stopping, not test results. Actual test metrics are:

{score}

Per-class precision, recall, F1 and support plus raw and true-class-normalised confusion matrices are under `outputs/audit/`. `outputs/predictions/error_analysis_examples.csv` contains genuine sampled correct and incorrect rows for qualitative review. No claim is made against an official benchmark because this is a separately sampled subset and different method.

## 6. Limitations and human verification
The output reflects one seeded sample and a lexical, reduced-dimensional representation; labels can retain ambiguity between substitutes and complements. Before submission, the student must inspect the visual figures and sampled errors, verify the data licence/citation and acquisition provenance, rerun the assignment profile on the intended machine, and write their own account of interpretation and decisions. No synthetic data or injected cleaning demo was used in this formal result.
""",encoding="utf-8")
    return checks
