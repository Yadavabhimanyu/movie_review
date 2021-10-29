import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

df = pd.read_csv(r'G:DS_data/IMDB_movies.csv')

X = df['review'][:30000]
y = df['sentiment'][:30000]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1111)

tfv = TfidfVectorizer(stop_words='english')
train_x_vector = tfv.fit_transform(X_train)
test_x_vector = tfv.transform(X_test)

log_reg = LogisticRegression()
log_reg.fit(train_x_vector, y_train)

print(log_reg.score(test_x_vector, y_test))

para = {'C': [0.1, 0.2, 0.3, 0.4, 0.5]}
log = LogisticRegression()
log_grid = GridSearchCV(log, para, cv=5)

log_grid.fit(train_x_vector, y_train)
print(log_grid.best_params_)
print(log_grid.score(test_x_vector, y_test))

# saving model to disk

pickle.dump(log_reg, open('model.pkl', 'wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict(tfv.transform(['A good movie'])))

pickle.dump(tfv,open('tfv.pkl' , 'wb'))