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
cv = TfidfVectorizer(ngram_range=(1, 3), stop_words=stop_words)
cv.fit(reviews_train_clean)
X = cv.transform(reviews_train_clean)
X_test = cv.transform(reviews_test_clean)

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
# split the train set into training data and validation data
X_train, X_val, y_train, y_val = train_test_split(
    X, target, train_size=0.75
)

accuracy = []
regularization = [0.01, 0.05, 0.25, 0.5, 1]
for c in regularization:
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    accuracy.append(accuracy_score(y_val, lr.predict(X_val)))
    print("Accuracy for C=%s: %s"
          % (c, accuracy_score(y_val, lr.predict(X_val))))


# train final model with highest c
highestC = [regularization[i] for i in range(len(regularization)) if accuracy[i] == max(accuracy)][0]
final_model = LogisticRegression(C = highestC)
final_model.fit(X, target)
y_pred = final_model.predict(X_test)
print("Final Accuracy: %s" % accuracy_score(target, y_pred))
print(str(highestC))
accuracy.append(accuracy_score(target, final_model.predict(X_test)))
regularization.append('testError C:' + str(highestC))
plt.plot(regularization, accuracy)
plt.savefig("ab3A (3 grams).pdf")

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