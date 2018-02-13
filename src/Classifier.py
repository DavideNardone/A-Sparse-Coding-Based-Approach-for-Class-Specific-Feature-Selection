from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
#add the need classifiers when using this class



class Classifier:

    def __init__(self, names=None, classifiers=None):

        self.cv_scores = {}

        #Default classifiers and parameters
        if names == None:

            self.names = [
                "KNN", "Logistic Regression", "SVM",
                "Decision Tree", "Random Forest", "AdaBoost"
            ]

            self.classifiers = [

                KNeighborsClassifier(n_neighbors=1),
                LogisticRegression(C=1e5),
                SVC(kernel="linear"),
                DecisionTreeClassifier(max_depth=5),
                RandomForestClassifier(max_depth=5, n_estimators=10),
                AdaBoostClassifier()
            ]

        else:
            self.names = names
            self.classifiers = classifiers

        for name in self.names:
            self.cv_scores[name] = []



    def train(self, X_train, y_train):

        for name, clf in zip(self.names, self.classifiers):

            # Training the algorithm using the selected predictors and target.
            clf.fit(X_train, y_train)

    def classify(self, X_test, y_test):

        # Record error for training and testing
        DTS = {}

        for name, clf in zip(self.names, self.classifiers):

            preds = clf.predict(X_test)

            dic_label = {
                name: preds
            }

            DTS.update(dic_label)

        return DTS