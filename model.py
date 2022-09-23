import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold,RandomizedSearchCV,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,hinge_loss,log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("G:/Chronic Kidney Disease classification/kidney_disease.csv")
df.drop([61], inplace=True)
print(df)

numerical_features = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sod", "pot", "hemo", "pcv", "rc", "sc", "wc"]
categorical_features = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]

# Data Cleaning---

gcol = df[["pcv", "rc", "wc"]]
for i in gcol:
    df[i].replace(to_replace='\t?', value=np.nan, inplace=True)
    df[i] = df[i].astype('float64')

df['classification'] = df["classification"].replace(to_replace="ckd\t", value="ckd", inplace=True)
df['dm'] = df["dm"].replace(to_replace=['\tno', '\tyes', ' yes'], value=['no', 'yes', 'yes'], inplace=True)
df['cad'] = df["cad"].replace(to_replace="\tno", value="no", inplace=True)


# Outlier handling via capping technique---

outlier_features = df[["age", "bp", "bgr", "bu", "sod", "pot", "hemo", "pcv", "rc", "sc", "wc"]]
for i in outlier_features:
    q1 = df[i].quantile(0.25)
    q3 = df[i].quantile(0.75)
    iqr = q3-q1
    lowerlimit = q1-(1.5*iqr)
    upperlimit = q3+(1.5*iqr)
    df[i] = np.where(df[i] < lowerlimit, lowerlimit, np.where(df[i] > upperlimit, upperlimit, df[i]))


# Dividing data into train and test ---

x = df.iloc[:, :24]
y = df["classification"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Feature Engineering (Imputing missing values, Encoding categorical variables, Scaling) ----

numerical_transformer = Pipeline(steps=
                                 [("imputer", KNNImputer(n_neighbors=5, weights="uniform", metric="nan_euclidean")),
                                  ("scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=
                                   [("encoder", OneHotEncoder(sparse=False, handle_unknown="ignore")),
                                    ("imputer", KNNImputer(n_neighbors=5, weights="uniform", metric="nan_euclidean"))])

preprocessor = make_column_transformer((numerical_features, numerical_transformer),
                                       (categorical_features, categorical_transformer),
                                       remainder='passthrough')


# Modelling ---

svc_model = SVC(kernel='linear', class_weight='balanced')
knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
gnb_model = GaussianNB()
logreg_model = LogisticRegression()
dtc_model = DecisionTreeClassifier()
rfc_model = RandomForestClassifier()

repository = [logreg_model, svc_model, knn_model, gnb_model, dtc_model, rfc_model]

for model in repository:
    clf = Pipeline(steps=[("preprocessor", preprocessor),
                          ("classifier", model)])
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(f'Model: {model}')
    print(f'Accuracy: {accuracy_score(y_pred,y_test)*100}')

