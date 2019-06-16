<p style="text-align:justify"><u><strong>load dataset:</strong></u></p>

<p style="text-align:justify">To prepare the dataset, load the downloaded data into a pandas dataframe&nbsp;containing two columns &ndash; text and label.</p>

<p style="text-align:justify">&nbsp;</p>

<p style="text-align:justify">&nbsp; load_data = dataset_load();</p>

<p style="text-align:justify">&nbsp; trainDF = load_data.load_cvs_dataset(<em>&quot;../corpus.csv&quot;</em>)</p>

<p style="text-align:justify">&nbsp; txt_label = trainDF[<em>&#39;label&#39;</em>]</p>

<p style="text-align:justify">&nbsp; txt_text = trainDF[<em>&#39;text&#39;</em>]</p>

<p style="text-align:justify">&nbsp;</p>

<p style="text-align:justify">this code segment&nbsp; found in trainmodel_write.py</p>

<p style="text-align:justify">&nbsp;</p>

<p style="text-align:justify">&nbsp; def&nbsp;<strong>load_cvs_dataset</strong>(<em>self</em>,dataset_path):</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp;#Set Random seed</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp;np.random.seed(500)</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp;# Add the Data using pandas</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp;Corpus = pd.read_csv(dataset_path,encoding=<em>&#39;latin-1&#39;</em>,error_bad_lines=False)</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</p>

<p style="text-align:justify">&nbsp; return&nbsp;Corpus</p>

<p style="text-align:justify">this code segment found in&nbsp;<strong>dataset_load.py</strong></p>

<p style="text-align:justify">&nbsp;</p>

<p style="text-align:justify"><u><strong>Text Feature Engineering:</strong></u></p>

<p style="text-align:justify">The next step is the feature engineering step. In this step, raw text data will be transformed into feature vectors and new features will be created using the existing dataset. We will implement the following different ideas in order to obtain relevant features from our dataset.</p>

<p style="text-align:justify">Count Vectors as features<br />
TF-IDF Vectors as features</p>

<ul>
	<li>
	<p style="text-align:justify">Word level</p>
	</li>
	<li>
	<p style="text-align:justify">N-Gram level</p>
	</li>
	<li>
	<p style="text-align:justify">Character level</p>
	</li>
</ul>

<p style="text-align:justify">Lets look at the implementation of these ideas in detail.</p>

<p style="text-align:justify"><u><strong>Count Vectors as features:</strong></u></p>

<p style="text-align:justify">Count Vector is a matrix notation of the dataset in which every row represents a document from the corpus, every column represents a term from the corpus, and every cell represents the frequency count of a particular term in a particular document.</p>

<p style="text-align:justify">&nbsp;</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; &nbsp; Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(txt_text,txt_label)</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; count_vect = CountVectorizer(analyzer=<em>&#39;word&#39;</em>)</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; count_vect.fit(txt_text)</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# transform the training and validation data using count vectorizer object</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Train_X_count =&nbsp; count_vect.transform(Train_X)</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Test_X_count =&nbsp; count_vect.transform(Test_X)&nbsp;&nbsp;</p>

<p style="text-align:justify"><u><strong>TF-IDF Vectors as features:</strong></u></p>

<p style="text-align:justify">TF-IDF score represents the relative importance of a term in the document and the entire corpus. TF-IDF score is composed by two terms: the first computes the normalized Term Frequency (TF), the second term is the Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents in the corpus divided by the number of documents where the specific term appears.</p>

<p style="text-align:justify">TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)<br />
IDF(t) = log_e(Total number of documents / Number of documents with term t in it)</p>

<p style="text-align:justify">TF-IDF Vectors can be generated at different levels of input tokens (words, characters, n-grams)</p>

<p style="text-align:justify"><strong>a. Word Level TF-IDF :</strong>&nbsp;Matrix representing tf-idf scores of every term in different documents<br />
<strong>b. N-gram Level TF-IDF :</strong>&nbsp;N-grams are the combination of N terms together. This Matrix representing tf-idf scores of N-grams<br />
<strong>c. Character Level TF-IDF :</strong>&nbsp;Matrix representing tf-idf scores of character level n-grams in the corpus</p>

<p style="text-align:justify"><strong>Word Level TF-IDF :&nbsp;</strong></p>

<p style="text-align:justify">&nbsp;</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; &nbsp; Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(txt_text, txt_label)</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;encoder = preprocessing.LabelEncoder()</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Train_Y = encoder.fit_transform(Train_Y)</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Test_Y = encoder.fit_transform(Test_Y)</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; tfidf_vect = TfidfVectorizer(analyzer=<em>&#39;word&#39;</em>,max_features=5000)</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; tfidf_vect.fit(txt_text)</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Train_X_Tfidf = tfidf_vect.transform(Train_X)</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Test_X_Tfidf = tfidf_vect.transform(Test_X)</p>

<p style="text-align:justify">This code segment fond in&nbsp;word_tf_idf.py</p>

<p style="text-align:justify"><strong>N-gram Level TF-IDF:</strong></p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; &nbsp; Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(txt_text, txt_label)</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; &nbsp;tfidf_vect_ngram = TfidfVectorizer(analyzer=<em>&#39;word&#39;</em>, ngram_range=(2,&nbsp;3), max_features=5000)</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; tfidf_vect_ngram.fit(txt_text)</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Train_X_ngram = tfidf_vect_ngram.transform(Train_X)</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Test_X_ngram = tfidf_vect_ngram.transform(Test_X)</p>

<p style="text-align:justify"><strong>This code segment fond in&nbsp;ngram_tf_idf.py</strong></p>

<p style="text-align:justify"><strong><strong>Character Level TF-IDF:</strong></strong></p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; &nbsp; Train_X, Test_X, Train_Y, Test_Y&nbsp; = model_selection.train_test_split(txt_text, txt_label)</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; tfidf_vect_ngram_chars = TfidfVectorizer(analyzer=<em>&#39;char&#39;</em>, token_pattern=<em>r&#39;\w{1,}&#39;</em>, ngram_range=(2,3), max_features=5000)</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; tfidf_vect_ngram_chars.fit(txt_text)</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Train_X_ngram_chars =&nbsp; tfidf_vect_ngram_chars.transform(Train_X)</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Test_X_ngram_chars =&nbsp; tfidf_vect_ngram_chars.transform(Test_X)</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</p>

<p style="text-align:justify"><u><strong>Model Training &amp; evaluate the performance of model:</strong></u></p>

<p style="text-align:justify">The final step in the text classification framework is to train a classifier using the features created in the previous step. There are many different choices of machine learning models which can be used to train a final model. We will implement following different classifiers for this purpose:</p>

<ol>
	<li>
	<p style="text-align:justify">Naive Bayes Classifier</p>
	</li>
	<li>
	<p style="text-align:justify">Linear Classifier</p>
	</li>
	<li>
	<p style="text-align:justify">Support Vector Machine</p>
	</li>
	<li>
	<p style="text-align:justify">Bagging Models</p>
	</li>
</ol>

<p style="text-align:justify">Lets implement these models and understand their details. The following function is a utility function which can be used to train a model. It accepts the classifier, feature_vector of training data, labels of training data and feature vectors of valid data as inputs. Using these inputs, the model is trained and accuracy score is computed.</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; def&nbsp;<strong>train_model</strong>(<em>self</em>,classifier, train_input,test_input, train_target, test_target, is_neural_net=False):</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; &nbsp;</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; classifier.fit(train_input, train_target)</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;predictions = classifier.predict(test_input)</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;if&nbsp;is_neural_net:</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; predictions = predictions.argmax(axis=-1</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;return&nbsp;accuracy_score(predictions, test_target)</p>

<p style="text-align:justify"><u><strong>Naive Bayes:</strong></u></p>

<p style="text-align:justify">Implementing a naive bayes model using sklearn implementation with different features</p>

<p style="text-align:justify">Naive Bayes is a classification technique based on Bayes&rsquo; Theorem with an assumption of independence among predictors. A Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; &nbsp;model_input = char_tf_idf().convert_feature(txt_text, txt_label)</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; &nbsp;naive = naive_bayes.MultinomialNB()</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; &nbsp;accuracy = Classifier().train_model(naive, model_input.get_train_input(), model_input.get_test_input(),&nbsp;</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; &nbsp;model_input.get_train_target(),</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; &nbsp;model_input.get_test_target())</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; &nbsp;print&nbsp;(<em>&quot;NB, char_tf_idf accuracy is : &quot;</em>, accuracy *&nbsp;100)</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; &nbsp;model_input = count_vectorizer().convert_feature(txt_text, txt_label)</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; &nbsp;naive = naive_bayes.MultinomialNB()</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; &nbsp;accuracy = Classifier().train_model(naive, model_input.get_train_input(), model_input.get_test_input(),&nbsp;model_input.get_train_target(), model_input.get_test_target())</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; &nbsp;print&nbsp;(<em>&quot;NB, count_vectorizer accuracy is : &quot;</em>, accuracy *&nbsp;100)</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; &nbsp;model_input = ngram_tf_idf().convert_feature(txt_text, txt_label)</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; &nbsp;naive = naive_bayes.MultinomialNB()</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; accuracy = Classifier().train_model(naive, model_input.get_train_input(), model_input.get_test_input(), model_input.get_train_target(), model_input.get_test_target())</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp;&nbsp;print&nbsp;(<em>&quot;NB, ngram_tf_idf accuracy is : &quot;</em>, accuracy *&nbsp;100)</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; model_input = word_tf_idf().convert_feature(txt_text, txt_label)</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp;&nbsp;naive = naive_bayes.MultinomialNB()</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp; accuracy = Classifier().train_model(naive, model_input.get_train_input(), model_input.get_test_input(), model_input.get_train_target(), model_input.get_test_target())</p>

<p style="text-align:justify">&nbsp; &nbsp; &nbsp;&nbsp;print&nbsp;(<em>&quot;NB, word_tf_idf accuracy is : &quot;</em>, accuracy *&nbsp;100)</p>

<p style="text-align:justify">&nbsp;&nbsp;</p>

<p style="text-align:justify">this code segment found in&nbsp;<strong>Naive_Bay_Clf.py</strong></p>

<p style="text-align:justify">&nbsp;</p>

<p style="text-align:justify">NB, char_tf_idf accuracy is :&nbsp; 81.28</p>

<p style="text-align:justify">NB, count_vectorizer accuracy is :&nbsp; 82.96</p>

<p style="text-align:justify">NB, ngram_tf_idf accuracy is :&nbsp; 81.92</p>

<p style="text-align:justify">NB, word_tf_idf accuracy is :&nbsp; 85.96000000000001</p>

<p style="text-align:justify">&nbsp;</p>

<p style="text-align:justify"><u><strong>Linear Classifier:</strong></u></p>

<p style="text-align:justify">Implementing a Linear Classifier (Logistic Regression)</p>

<p style="text-align:justify">Logistic regression measures the relationship between the categorical dependent variable and one or more independent variables by estimating probabilities using a logistic/sigmoid function.</p>

<p style="text-align:justify">#&nbsp; Build Text Classification Model and Evaluating the Model</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp; naive = linear_model.LogisticRegression()</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp; accuracy = Classifier().train_model(naive, model_input.get_train_input(), model_input.get_test_input(), model_input.get_train_target(), model_input.get_test_target())</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;print&nbsp;(<em>&quot;Linear_Clf, count_vectorizer accuracy is : &quot;</em>, accuracy *&nbsp;100)</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;# Text feature engineering</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp; model_input = ngram_tf_idf().convert_feature(txt_text, txt_label)</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;# Text feature engineering</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;#&nbsp; Build Text Classification Model and Evaluating the Model</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp; naive = linear_model.LogisticRegression()</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp; accuracy = Classifier().train_model(naive, model_input.get_train_input(), model_input.get_test_input(), model_input.get_train_target(), model_input.get_test_target())</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;print&nbsp;(<em>&quot;Linear_Clf, ngram_tf_idf accuracy is : &quot;</em>, accuracy *&nbsp;100)</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;# Text feature engineering</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp; model_input = word_tf_idf().convert_feature(txt_text, txt_label)</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;# Text feature engineering</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;#&nbsp; Build Text Classification Model and Evaluating the Model</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp; naive = linear_model.LogisticRegression()</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp; accuracy = Classifier().train_model(naive, model_input.get_train_input(), model_input.get_test_input(), model_input.get_train_target(), model_input.get_test_target())</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;print&nbsp;(<em>&quot;Linear_Clf, word_tf_idf accuracy is : &quot;</em>, accuracy *&nbsp;100)</p>

<p style="text-align:justify">&nbsp;</p>

<p style="text-align:justify">Linear_Clf, char_tf_idf accuracy is :&nbsp; 84.36</p>

<p style="text-align:justify">Linear_Clf, count_vectorizer accuracy is :&nbsp; 85.92</p>

<p style="text-align:justify">Linear_Clf, ngram_tf_idf accuracy is :&nbsp; 82.64</p>

<p style="text-align:justify">Linear_Clf, word_tf_idf accuracy is :&nbsp; 87.4</p>

<p style="text-align:justify">&nbsp;</p>

<p style="text-align:justify"><u><strong>&nbsp;SVM Model:</strong></u></p>

<p style="text-align:justify">Support Vector Machine (SVM) is a supervised machine learning algorithm which can be used for both classification or regression challenges. The model extracts a best possible hyper-plane / line that segregates the two classes.</p>

<p style="text-align:justify">#&nbsp; Build Text Classification Model and Evaluating the Model</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp; naive=svm.SVC()</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp; accuracy = Classifier().train_model(naive,model_input.get_train_input(),model_input.get_test_input(), model_input.get_train_target(), model_input.get_test_target())</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;print&nbsp;(<em>&quot;Svm_clf, ngram_tf_idf accuracy is : &quot;</em>, accuracy*100)</p>

<p style="text-align:justify">&nbsp;</p>

<p style="text-align:justify">Svm_clf, ngram_tf_idf accuracy is :&nbsp; 51.76</p>

<p style="text-align:justify">&nbsp;</p>

<p style="text-align:justify"><u><strong>Random Forest Model:</strong></u></p>

<p style="text-align:justify">Random Forest models are a type of ensemble models, particularly bagging models. They are part of the tree based model family.</p>

<p style="text-align:justify">#&nbsp; Build Text Classification Model and Evaluating the Model</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp; naive=ensemble.RandomForestClassifier()</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp; accuracy = Classifier().train_model(naive,model_input.get_train_input(),model_input.get_test_input(), model_input.get_train_target(), model_input.get_test_target())</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;print&nbsp;(<em>&quot;RandomForest_Clf, count_vectorizer accuracy is : &quot;</em>, accuracy*100)</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;#Text feature engineering</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp; model_input=word_tf_idf().convert_feature(clear_txt,txt_label)</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;#Text feature engineering</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;#&nbsp; Build Text Classification Model and Evaluating the Model</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp; naive=ensemble.RandomForestClassifier()</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp; accuracy = Classifier().train_model(naive,model_input.get_train_input(),model_input.get_test_input(), model_input.get_train_target(), model_input.get_test_target())</p>

<p style="text-align:justify">&nbsp;&nbsp;&nbsp;&nbsp;print&nbsp;(<em>&quot;RandomForest_Clf, word_tf_idf accuracy is : &quot;</em>, accuracy*100)</p>

<p style="text-align:justify">&nbsp;</p>

<p style="text-align:justify">&nbsp;</p>

<p style="text-align:justify">RandomForest_Clf, count_vectorizer accuracy is :&nbsp; 77.84</p>

<p style="text-align:justify">RandomForest_Clf, word_tf_idf accuracy is :&nbsp; 78.52</p>
