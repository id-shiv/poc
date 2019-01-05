"""
Data Science modules
@author : Shiva Prasad
E-mail : shiv.id@icloud.com

"""

# import required modules
import logging
import warnings

import pandas as pd

warnings.simplefilter(action='ignore', category=Warning)
logging.basicConfig(filename='data-science-modules/module.log',
                    format='[%(levelname)s, %(asctime)s, %(filename)s, '
                           '%(module)s, %(funcName)s, line#%(lineno)s]%(message)s',
                    level=logging.DEBUG)


class DataScience(object):
    """
    # TEMPLATE ---------------------------------------
    Supervised
        Classification
        Regression
    Un-Supervised
        Clustering

    # ------------------------------------------------
    # Understand the problem
    Output : Success metric (e.g. Mean Absolute Error)
    # ------------------------------------------------

    # ------------------------------------------------
    # Collect data
    Input : urls, csvs
    Methods : Web-Scraping, Pandas manipulation
    Output : Unstructured Data
    # ------------------------------------------------

    # ------------------------------------------------
    # Import data
    Input : Unstructured Data
    Output : DataFrame + Corpus (if text data)
    # ------------------------------------------------

    # ------------------------------------------------
    # Process data
    Actions :
        Handle missing values,
        Dummy variables for categorical data,
        Remove redundant samples
        Scale feature values of continuous data
    Output : DataFrame + Corpus (if text data)
    # ------------------------------------------------

    # ------------------------------------------------
    # Analyze data
    Input : DataFrame + Corpus (if text data)
    Perform EDA (Exploratory Data Analysis)
    # ------------------------------------------------

    # ------------------------------------------------
    # Clean data
    Actions : Standardize, text to vector
    Next step : Re-analyze data
    # ------------------------------------------------

    # ------------------------------------------------
    # Model data

    1. Regression Algorithms

    Ordinary Least Squares Regression (OLSR)
    Linear Regression
    Logistic Regression
    Stepwise Regression
    Multivariate Adaptive Regression Splines (MARS)
    Locally Estimated Scatterplot Smoothing (LOESS)

    # --------------------
    2. Instance-based Algorithms

    k-Nearest Neighbour (kNN)
    Learning Vector Quantization (LVQ)
    Self-Organizing Map (SOM)
    Locally Weighted Learning (LWL)

    # --------------------
    3. Regularization Algorithms

    Ridge Regression
    Least Absolute Shrinkage and Selection Operator (LASSO)
    Elastic Net
    Least-Angle Regression (LARS)
    4. Decision Tree Algorithms

    Classification and Regression Tree (CART)
    Iterative Dichotomiser 3 (ID3)
    C4.5 and C5.0 (different versions of a powerful approach)
    Chi-squared Automatic Interaction Detection (CHAID)
    Decision Stump
    M5
    Conditional Decision Trees

    # --------------------
    5. Bayesian Algorithms

    Naive Bayes
    Gaussian Naive Bayes
    Multinomial Naive Bayes
    Averaged One-Dependence Estimators (AODE)
    Bayesian Belief Network (BBN)
    Bayesian Network (BN)

    # --------------------
    6. Clustering Algorithms

    k-Means
    k-Medians
    Expectation Maximisation (EM)
    Hierarchical Clustering

    # --------------------
    7. Association Rule Learning Algorithms

    Apriori algorithm
    Eclat algorithm

    # --------------------
    8. Artificial Neural Network Algorithms

    Perceptron
    Back-Propagation
    Hopfield Network
    Radial Basis Function Network (RBFN)

    # --------------------
    9. Deep Learning Algorithms

    Deep Boltzmann Machine (DBM)
    Deep Belief Networks (DBN)
    Convolutional Neural Network (CNN)
    Stacked Auto-Encoders

    # --------------------
    10. Dimensionality Reduction Algorithms

    Principal Component Analysis (PCA)
    Principal Component Regression (PCR)
    Partial Least Squares Regression (PLSR)
    Sammon Mapping
    Multidimensional Scaling (MDS)
    Projection Pursuit
    Linear Discriminant Analysis (LDA)
    Mixture Discriminant Analysis (MDA)
    Quadratic Discriminant Analysis (QDA)
    Flexible Discriminant Analysis (FDA)

    # --------------------
    11. Ensemble Algorithms

    Boosting
    Bootstrapped Aggregation (Bagging)
    AdaBoost
    Stacked Generalization (blending)
    Gradient Boosting Machines (GBM)
    Gradient Boosted Regression Trees (GBRT)
    Random Forest

    # --------------------
    12. Other Algorithms

    Computational intelligence (evolutionary algorithms, etc.)
    Computer Vision (CV)
    Natural Language Processing (NLP)
    Recommender Systems
    Reinforcement Learning
    Graphical Models

    # --------------------
    Decide : is data labelled? is data categorical or continuous?
    # ------------------------------------------------

    # ------------------------------------------------
    # Report data product
    Input : Model output
    Output : Predictions, Decisions, Visualizations
    # ------------------------------------------------
    """

    class Collect(object):
        @staticmethod
        def scrape_web(url, tag='p'):
            from bs4 import BeautifulSoup
            import requests

            text = ""

            r = requests.get(url)
            data = r.text
            soup = BeautifulSoup(data)
            for strong_tag in soup.find_all(tag):
                try:
                    if strong_tag.text.strip():
                        text = text + "\n" + strong_tag.text.strip()
                except:
                    pass

            logging.debug("\n" + "-" * 50 + url + " " + "-" * 50 + "\n" + str(text))
            return text

    class Import(object):
        @staticmethod
        def file2dataset(file, header=False):
            file_extension = file.split('.')[-1]
            dataset = pd.DataFrame()
            if file_extension == 'csv':
                logging.info(file + " is csv")
                if not header:
                    dataset = pd.read_csv(file, header=None)
                else:
                    dataset = pd.read_csv(file)
            elif file_extension == 'tsv':
                logging.info(file + " is tsv")
                if not header:
                    dataset = pd.read_csv(file, header=None, sep='\t')
                else:
                    dataset = pd.read_csv(file, sep='\t')
            elif file_extension == 'pdf':
                import textract
                text = textract.process(file)
                dataset[0][0] = text
            else:
                logging.error("File extension not supported : " + file)

            logging.debug("\n" + "-" * 50 + "dataframe" + "-" * 50 + "\n" + str(dataset.head()))
            return dataset

        @staticmethod
        def news_groups():
            from sklearn.datasets import fetch_20newsgroups
            return fetch_20newsgroups(shuffle=True, random_state=1,
                                      remove=('headers', 'footers', 'quotes'))

    class PreProcess(object):
        """
        Feature extraction
        Data Pre-processing
        Imputation of missing values
        Dimensionality reduction
        Kernel approximation
        """
        class ProcessText(object):

            @staticmethod
            def sentences_get(text):
                from nltk.tokenize import sent_tokenize
                return sent_tokenize(text)

            def tokens_get(self, text, remove_punctuation=True):
                if not remove_punctuation:
                    from nltk.tokenize import word_tokenize
                    return word_tokenize(text)
                else:
                    return self.__tokens_remove_punctuations(text)

            @staticmethod
            def __tokens_remove_punctuations(text):
                from nltk.tokenize import RegexpTokenizer

                tokenizer = RegexpTokenizer(r'\w+')
                return tokenizer.tokenize(text)

            @staticmethod
            def paragraphs_get(text):
                from nltk.tokenize import blankline_tokenize
                return blankline_tokenize(text)

            @staticmethod
            def words_stem(tokens):
                """
                convert all words to root of that word
                :param tokens:
                :return:
                """
                from nltk.stem import PorterStemmer, LancasterStemmer
                stemmer = LancasterStemmer()
                return [stemmer.stem(word) for word in tokens]

            @staticmethod
            def words_remove_stop_words(tokens):
                """
                :param tokens:
                :return:
                """
                # import nltk
                # nltk.download('stopwords')
                # from nltk.corpus import stopwords
                stopwords = ["a", "the", "in", "for", "of", "to",
                             "this", "and", "is"]
                return [word for word in tokens if word not in stopwords]

            def ngrams_get(self, text):
                from nltk.util import ngrams
                return list(ngrams(self.tokens_get(text), 3))

            @staticmethod
            def tokens_count_total(tokens):
                return len(tokens)

            def tokens_count_unique(self, tokens):
                return len(self.__tokens_frequency_get(tokens))

            @staticmethod
            def __tokens_frequency_get(tokens):
                from nltk.probability import FreqDist
                fdist = FreqDist()
                for token in tokens:
                    fdist[token.lower()] += 1
                return fdist

            def tokens_occurance_gt_n(self, tokens, n=1):
                """
                :param tokens:
                :param n:
                :return:
                """
                token_freq = self.__tokens_frequency_get(tokens)
                return [token for token in token_freq if token_freq[token] > n]

            @staticmethod
            def pos_tags_get(tokens):
                import nltk
                return nltk.pos_tag(tokens)

            def ner_tags_get(self, tokens):
                from nltk import ne_chunk
                return ne_chunk(self.pos_tags_get(tokens))

            def chunks_get(self, tokens, grammer = r"NP : {<DT>?<JJ>*<NN>}"):
                import nltk
                chunk_parser = nltk.RegexpParser(grammer)
                return chunk_parser.parse(self.pos_tags_get(tokens))

            def most_common(self, tokens, n=1):
                fdist = self.__tokens_frequency_get(tokens)
                return fdist.most_common(n)

        @staticmethod
        def print_topics(model, feature_names, n_top_words):
            for topic_idx, topic in enumerate(model.components_):
                message = "Topic #%d: " % topic_idx
                message += " ".join([feature_names[i]
                                     for i in topic.argsort()[:-n_top_words - 1:-1]])
                print(message)
            print()

    class Model(object):
        def __init__(self, X, y=None):
            self.X = X
            self.y = y
            self.fitted = None
            self.model = None

        def fit(self, model_type, model_name):
            if model_type == 'regression' and model_name == 'random_forest':
                self.model, self.fitted = self.__regression_random_forest()
            elif model_type == 'clustering' and model_name == 'lda':
                self.model, self.fitted = self.__dim_reduction_latent_dirichlet_allocation()
            elif model_type == 'clustering' and model_name == 'nmf':
                self.model, self.fitted = self.__dim_reduction_nmf()
            elif model_type == 'vectorize' and model_name == 'tfid':
                self.model, self.fitted = self.__vectorize('tfid')
            elif model_type == 'vectorize' and model_name == 'count':
                self.model, self.fitted = self.__vectorize('count')

        def predict(self, X_pred):
            return self.fit, self.model.predict([X_pred])

        def __vectorize(self, vectorizer, n=20):
            model = None
            if vectorizer == 'tfid':
                from sklearn.feature_extraction.text import TfidfVectorizer
                model = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n, stop_words='english')
            elif vectorizer == 'count':
                from sklearn.feature_extraction.text import CountVectorizer
                model = CountVectorizer(max_df=0.95, min_df=2, max_features=n, stop_words='english')

            return model, model.fit_transform(self.X)

        def __regression_linear(self):
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            return model, model.fit(self.X, self.y)

        def __regression_polynomial(self):
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import PolynomialFeatures
            poly_reg = PolynomialFeatures(degree=4)
            X_poly = poly_reg.fit_transform(self.X)
            poly_reg.fit(X_poly, self.y)
            model = LinearRegression()
            return model, model.fit(X_poly, self.y)

        def __regression_support_vectors(self):
            from sklearn.svm import SVR
            model = SVR(kernel='rbf')
            return model, model.fit(self.X, self.y)

        def __regression_decision_tree(self):
            from sklearn.tree import DecisionTreeRegressor
            model = DecisionTreeRegressor(random_state=0)
            return model, model.fit(self.X, self.y)

        def __regression_random_forest(self):
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=300, random_state=0)
            return model, model.fit(self.X, self.y)

        def __classification_logistic_regression(self):
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=0)
            return model, model.fit(self.X, self.y)

        def __classification_knn(self):
            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
            return model, model.fit(self.X, self.y)

        def __classification_support_vectors(self):
            from sklearn.svm import SVC
            model = SVC(kernel='rbf', random_state=0)
            return model, model.fit(self.X, self.y)

        def __classification_naive_bayes(self):
            from sklearn.naive_bayes import GaussianNB
            model = GaussianNB()
            return model, model.fit(self.X, self.y)

        def __classification_decision_tree(self):
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(criterion='entropy', random_state=0)
            return model, model.fit(self.X, self.y)

        def __classification_random_forest(self):
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
            return model, model.fit(self.X, self.y)

        def __dim_reduction_latent_dirichlet_allocation(self):
            from sklearn.decomposition import LatentDirichletAllocation
            model = LatentDirichletAllocation(n_topics=10, max_iter=5,
                                              learning_method='online', learning_offset=5, random_state=0)
            return model, model.fit(self.X)

        def __dim_reduction_nmf(self, n=20):
            """
            Non-negative Matrix Factorization
            :return:
            """
            from sklearn.decomposition import NMF
            model = NMF(n_components=n, random_state=1, alpha=.1, l1_ratio=.5)
            return model, model.fit(self.X)

    class Evaluate(object):
        @staticmethod
        def evaluate():
            print('evaluate')

    class Examples(object):
        @staticmethod
        def topic_modelling_news_group():
            import_obj = DataScience.Import()
            dataset = import_obj.news_groups()
            dataset = dataset.data[:20]

            model_obj = DataScience.Model(dataset)

            model_obj.fit('vectorize', 'count')
            vectorized = model_obj.fitted

            feature_names = model_obj.model.get_feature_names()

            model_obj = DataScience.Model(vectorized)
            model_obj.fit('clustering', 'lda')

            process = DataScience.Transform()
            process.print_topics(model_obj.model, feature_names, 20)

        @staticmethod
        def deep_learning_tensor_flow_iris():
            import tensorflow as tf
            import numpy as np

            print(tf.__version__)

            from tensorflow.contrib.learn.python.learn.datasets import base

            # Data files
            IRIS_TRAINING = "data-science-modules/data-sets/Wine.csv"
            IRIS_TEST = "data-science-modules/data-sets/Wine.csv"

            # Load datasets.
            training_set = base.load_csv_with_header(filename=IRIS_TRAINING,
                                                     features_dtype=np.float32,
                                                     target_dtype=np.int)
            test_set = base.load_csv_with_header(filename=IRIS_TEST,
                                                 features_dtype=np.float32,
                                                 target_dtype=np.int)

            # Specify that all features have real-value data
            feature_name = "flower_features"
            feature_columns = [tf.feature_column.numeric_column(feature_name,
                                                                shape=[13])]
            classifier = tf.estimator.LinearClassifier(
                feature_columns=feature_columns,
                n_classes=3,
                model_dir="/tmp/iris_model")

            def input_fn(dataset):
                def _fn():
                    features = {feature_name: tf.constant(dataset.data)}
                    label = tf.constant(dataset.target)
                    return features, label

                return _fn

            # Fit model.
            classifier.train(input_fn=input_fn(training_set),
                             steps=1000)
            print('fit done')

            # Evaluate accuracy.
            accuracy_score = classifier.evaluate(input_fn=input_fn(test_set),
                                                 steps=100)["accuracy"]
            print('\nAccuracy: {0:f}'.format(accuracy_score))

            # Export the model for serving
            feature_spec = {'flower_features': tf.FixedLenFeature(shape=[4], dtype=np.float32)}

            serving_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

            classifier.export_savedmodel(export_dir_base='/tmp/iris_model' + '/export',
                                         serving_input_receiver_fn=serving_fn)


if __name__ == "__main__":
    # print(DataScience.__doc__)
    exampl = DataScience.Examples()
    exampl.deep_learning_tensor_flow_iris()




