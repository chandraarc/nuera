from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import  GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import apriori, association_rules

def get_data_model(model_name,train_data, result_column, additional_param):

   y = train_data.loc[:, result_column]
   x = train_data.drop(columns=[result_column])
   model = ''
   if(model_name=='LinearRegression'):
      model = LinearRegression()
      model.fit(x.values, y.values)

   if (model_name == 'OneVsRestClassifier'):
      model = OneVsRestClassifier(SVC())
      model.fit(x.values, y.values)

   if (model_name == 'KNeighborsClassifier'):
      model = KNeighborsClassifier(additional_param['n_neighbors'])
      model.fit(x.values, y.values)

   if (model_name == 'PolynomialRegression'):
      poly_features = PolynomialFeatures(degree=additional_param['dgree'])
      x = poly_features.fit_transform(x)
      model = LinearRegression()
      model.fit(x.values, y.values)

   if (model_name == 'DecisionTreeRegressor'):
      model = DecisionTreeRegressor()
      model.fit(x.values, y.values)

   if (model_name == 'RandomForestRegressor'):
      model = RandomForestRegressor()
      model.fit(x.values, y.values)

   if (model_name == 'LogisticRegression'):
      model = LogisticRegression(random_state=additional_param['random_state'])
      model.fit(x.values, y.values)

   if (model_name == 'GaussianNB'):
      model = GaussianNB()
      model.fit(x.values, y.values)

   if (model_name == 'SVC'):
      model = SVC(kernel = additional_param['linear'],gamma = additional_param['scale'], shrinking = additional_param['shrinking'])
      model.fit(x.values, y.values)

   if (model_name == 'PCA'):
      model = PCA(n_components=additional_param['n_components'])
      model.fit_transform(train_data)

   if (model_name == 'SVC'):
      model = KMeans(init=additional_param['init'],
                      n_clusters=additional_param['additional_param'],
                      n_init=additional_param['n_init'], random_state=additional_param['random_state'])
      model.fit(train_data)

   if (model_name == 'fpgrowth'):
      model = fpgrowth(train_data, min_support=additional_param['min_support'], use_colnames=additional_param['use_colnames'])

   if (model_name == 'apriori'):
      model = apriori(train_data, min_support=additional_param['min_support'], use_colnames=additional_param['use_colnames'])

   return model;

