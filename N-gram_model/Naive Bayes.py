import matplotlib.pyplot as plt
# reference source https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184
# reference source https://towardsdatascience.com/sentiment-analysis-with-python-part-2-4f71e7bde59a
# read the file
reviews_train = []
for line in open('../Final project/movie_data/full_train.txt', 'r'):
    reviews_train.append(line.strip())

reviews_test = []
for line in open('../Final project/movie_data/full_test.txt', 'r'):
    reviews_test.append(line.strip())

# clean and preprocess
import re
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")


def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]

    return reviews


reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)

# nomorlization
def get_lemmatized_text(corpus):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

reviews_train_clean = get_lemmatized_text(reviews_train_clean)
reviews_test_clean = get_lemmatized_text(reviews_test_clean)

# vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
stop_words = ['in', 'of', 'at', 'a', 'the']
tfidf_vectorizer = CountVectorizer(binary=False, ngram_range=(1, 3), stop_words=stop_words)
tfidf_vectorizer.fit(reviews_train_clean)
X = tfidf_vectorizer.transform(reviews_train_clean)
X_test = tfidf_vectorizer.transform(reviews_test_clean)



# classifier find c
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = [1 if i < 12500 else 0 for i in range(25000)]
final_model = MultinomialNB()
final_model.fit(X, target)
y_pred = final_model.predict(X_test)
print("Final Accuracy: %s" % accuracy_score(target, y_pred))

from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_pred, target).ravel()
print(str((tn, fp, fn, tp)))


# top positive negative word
feature_to_coef = {
    word: coef for word, coef in zip(
        cv.get_feature_names(), final_model.coef_[0]
    )
}
for best_positive in sorted(
        feature_to_coef.items(),
        key=lambda x: x[1],
        reverse=True)[:5]:
    print(best_positive)


for best_negative in sorted(
        feature_to_coef.items(),
        key=lambda x: x[1])[:5]:
    print(best_negative)