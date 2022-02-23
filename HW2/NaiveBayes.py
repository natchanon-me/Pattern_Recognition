import numpy as np
import pandas as pd

path = 'https://raw.githubusercontent.com/ekapolc/pattern_2022/main/HW/HW02/hr-employee-attrition-with-null.csv'
df = pd.read_csv(path, index_col=[0])

# Clean data
def Cleaning(df):
    df.loc[df["Attrition"] == 'No', "Attrition"] = 0.0
    df.loc[df["Attrition"] == 'Yes', "Attrition"] = 1.0
    df['Attrition'] = df['Attrition'].astype('float64')
    
    # convert each categorical to numerical
    cats = []
    for col in df.columns:
        if str(df[col].dtype) != 'float64':
            cats.append(str(col))
    for col in cats:
        df[col] = pd.Categorical(df[col]).codes
        
        df.loc[df[col] == -1, col] = np.nan
    # Drop EmployeeNumber & EmployeeCount
    df = df.drop(columns = ['EmployeeNumber', 'EmployeeCount'])
    
    return df

df = Cleaning(df)

def gen_test(df):

    index_yes = df[df['Attrition'] == 1].index
    index_no = df[df["Attrition"] == 0].index
    
    yes_10 = np.random.choice(len(index_yes), size=int(np.ceil(len(index_yes)*0.1)), replace=False)
    no_10 = np.random.choice(len(index_no), size=int(np.ceil(len(index_no)*0.1)), replace=False)
    
    test_index = []
    test_index.extend(np.array(index_yes[yes_10]))
    test_index.extend(np.array(index_no[no_10]))
    test_df = df.iloc[test_index]
    train_df = df.drop(index=test_index)
    return train_df, test_df

train_df, test_df = gen_test(df)

def process_discretize(train_df, test_df):
    decretize_list = []
    non_decretize = []
    for feat in df.columns:
        if feat == "Attrition" or (train_df[feat].nunique()==1):
            continue;
        if df[feat].nunique() > 10:
            decretize_list.append(feat)
        else:
            non_decretize.append(feat)
    return decretize_list, non_decretize

decretize_list, non_decretize = process_discretize(train_df, test_df)

def get_classified_df(train_df, test_df):    
    train_leave_df = train_df[train_df["Attrition"] == 1]
    train_stay_df = train_df[train_df["Attrition"] == 0]
    label = np.array(test_df["Attrition"])
    return train_leave_df, train_stay_df, label

train_leave_df, train_stay_df, label = get_classified_df(train_df, test_df)

# get Distribution form histogram
def get_binslist(df, feat, bins=10):
    x = df[~np.isnan(df[feat])][feat]
    bin_list = np.arange(min(x), max(x), (max(x)-min(x))/bins)
    bin_list[0] = -np.inf
    bin_list = np.append(bin_list, np.inf)
    digitized = np.digitize(x, bin_list)
    return bin_list, np.bincount(digitized)/len(x)

def get_problist(df, feat, bins_list):
    x = df[~np.isnan(df[feat])][feat]
    digitized = np.digitize(x, bins_list)
    prob_list = np.bincount(digitized)/len(x)
    return np.append(prob_list, np.zeros(len(bins_list)))

def get_digitized(x, bins_list):
    digitized = np.digitize(x, bins_list)
    return digitized

def get_prob(digitized, prob_list):
    return prob_list[digitized]

# Evaluation Metric
def confusion_metric(y_true, y_pred):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(y_true)):
        if((y_true[i]==1) and (y_true[i]==y_pred[i])):
            tp += 1
        elif((y_true[i]==0) and (y_true[i]!=y_pred[i])):
            fp += 1
        elif((y_true[i]==1) and (y_true[i]!=y_pred[i])):
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn

def get_metric(y_true, y_pred):
    assert len(y_true) == len(y_pred), "Both label must be the same length."
    tp, fp, fn, tn = confusion_metric(y_true, y_pred)
    accuracy = (tp+tn)/(len(y_true))
    precision = tp/(tp+fp+1e-9)
    recall = tp/(tp+fn+1e-9)
    f1 = 2*((recall*precision)/(recall+precision+1e-9))
    return accuracy, precision, recall, f1

# Naive bayes model
def Naive_bayes(threshold=0, set_bins = 10):
    prediction = np.zeros((len(test_df)))
    for lists in [decretize_list, non_decretize]:
        for feat in lists:
            if feat in non_decretize:
                bins = train_df[feat].nunique()
            else:
                bins = set_bins;
            bin_list_all, prob_list_all = get_binslist(train_df, feat, bins)
        # Leave side (Attrition==1) calculation
            prob_list = get_problist(train_leave_df, feat, bin_list_all)
            prob_list[0] = 1 # when log is applied, NaN term become zero that means not take into account.
            dg = get_digitized(test_df[feat], bin_list_all)
            dg[dg>(len(bin_list_all)-1)]=0 # NaN become index 0 of prob
            #dg = dg.reshape([len(dg), 1])
            p = get_prob(dg, prob_list)
            p += 1e-6
            log_p_leave = np.log(p)

        # Stay side (Attrition==0) calculation
            prob_list = get_problist(train_stay_df, feat, bin_list_all)
            prob_list[0] = 1 # when log is applied, NaN term become zero that means not take into account.
            dg = get_digitized(test_df[feat], bin_list_all)
            dg[dg>(len(bin_list_all)-1)]=0 # NaN become index 0 of prob
            #dg = dg.reshape([len(dg), 1])
            p = get_prob(dg, prob_list)
            p += 1e-6
            log_p_stay = np.log(p)
            feature_likelihood = log_p_leave-log_p_stay
        # Sum over features
            prediction += feature_likelihood

    # Prior calculation
    prior = np.array(train_df["Attrition"].value_counts()/len(train_df))
    log_prior_leave = np.log(prior[1]) # prior = 0.161
    log_prior_stay = np.log(prior[0])# prior = 0.839
    log_prior = log_prior_leave-log_prior_stay
    prediction += log_prior

    prediction[prediction >= threshold] = 1
    prediction[prediction < threshold] = 0
    
    return prediction

pred = Naive_bayes(threshold=-0.8, set_bins=10)
acc, precision, recall, f1 = get_metric(label, pred)

print("\n")
print("############### NAIVE BAYES PREDICTION ###############\n")
print("Evaluate Naive Bayes . . . ")
print("Accuracy : ", acc)
print("Precision : ", precision)
print("Recall : ", recall)
print("F1 Score : ", f1, "\n")

print("############### BASELINE PREDICTION ###############\n")
# Random choice
rng_pred = np.random.choice(a=[0,1], size=len(test_df), p=[0.5,0.5])
rng_pred

acc, prec, rec, f1 = get_metric(label, rng_pred)
print("Random choice evaluation metrics : . . .")
print("Accuracy : ", acc)
print("Precision : ", prec)
print("Recall : ", rec)
print("F1 Score : ", f1, "\n")

# Majority Rule
major_pred = np.zeros(len(test_df))
major_pred
acc, prec, rec, f1 = get_metric(label, rng_pred)
print("Majority Rule evaluation metrics : . . .")
print("Accuracy : ", acc)
print("Precision : ", prec)
print("Recall : ", rec)
print("F1 Score : ", f1)


print("\nShuffle Datasets . . .")
# Shuffle for 10 times
n = 10
all_acc = []
all_f1 = []
for i in range(n):
    train_df, test_df = gen_test(df)
    decretize_list, non_decretize = process_discretize(train_df, test_df)
    train_leave_df, train_stay_df, label = get_classified_df(train_df, test_df)
    pred = Naive_bayes(threshold=-0.8, set_bins=10)
    acc, precision, recall, f1 = get_metric(label, pred)
    print("Setting ", i+1, f"   : Accuracy = {acc:.3f} , F1 = {f1:.3f}")
    all_acc.append(acc)
    all_f1.append(f1)

print("\n")
print(f"Mean accuracy for 10 shuffles is {np.mean(all_acc):.4f}")
print(f"Mean F1 score for 10 shuffles is {np.mean(all_f1):.4f}")
print("\n")

