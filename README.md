<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><strong><u>load dataset</u></strong></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:#595858">To prepare the dataset, load the downloaded data into a pandas dataframe</span> <span style="color:#595858">containing two columns &ndash; text and label.</span></span></span></p>

<p style="margin-left:0in; margin-right:0in">&nbsp;</p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp; load_data = dataset_load();</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp; trainDF = load_data.load_cvs_dataset(</span><em><span style="color:#00aa00">&quot;../corpus.csv&quot;</span></em><span style="color:black">)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp; txt_label = trainDF[</span><em><span style="color:#00aa00">&#39;label&#39;</span></em><span style="color:black">]</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp; txt_text = trainDF[</span><em><span style="color:#00aa00">&#39;text&#39;</span></em><span style="color:black">]</span></span></span></p>

<p style="margin-left:0in; margin-right:0in">&nbsp;</p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">this code segment&nbsp; found in trainmodel_write.py</span></span></p>

<p style="margin-left:0in; margin-right:0in">&nbsp;</p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:blue">&nbsp; def</span><span style="color:black"> <strong>load_cvs_dataset</strong>(<em>self</em>,dataset_path):</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:silver">&nbsp; &nbsp; &nbsp;#Set Random seed</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp; &nbsp; &nbsp;np.random.seed(</span><span style="color:maroon">500</span><span style="color:black">)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:silver">&nbsp; &nbsp; &nbsp;# Add the Data using pandas</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp; &nbsp; &nbsp;Corpus = pd.read_csv(dataset_path,encoding=</span><em><span style="color:#00aa00">&#39;<u>latin</u>-1&#39;</span></em><span style="color:black">,error_bad_lines=</span><span style="color:blue">False</span><span style="color:black">)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:blue">&nbsp; return</span><span style="color:black"> Corpus</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">this code segment found in <strong><span style="background-color:#ffff96">dataset_load.py</span></strong></span></span></span></p>

<p style="margin-left:0in; margin-right:0in">&nbsp;</p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><strong><u>Text Feature Engineering:</u></strong></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><span style="color:#595858">The next step is the feature engineering step. In this step, raw text data will be transformed into feature vectors and new features will be created using the existing dataset. We will implement the following different ideas in order to obtain relevant features from our dataset.</span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><span style="color:#595858">Count Vectors as features<br />
TF-IDF Vectors as features</span></span></span></span></p>

<ul>
	<li>
	<p><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><span style="color:#595858">Word level</span></span></span></span></p>
	</li>
	<li>
	<p><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><span style="color:#595858">N-Gram level</span></span></span></span></p>
	</li>
	<li>
	<p><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><span style="color:#595858">Character level</span></span></span></span></p>
	</li>
</ul>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><span style="color:#595858">Lets look at the implementation of these ideas in detail.</span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><strong><u><span style="color:#333333">Count Vectors as features</span></u></strong></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><span style="color:#595858">Count Vector is a matrix notation of the dataset in which every row represents a document from the corpus, every column represents a term from the corpus, and every cell represents the frequency count of a particular term in a particular document.</span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in">&nbsp;</p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:#ffff96"><span style="color:black">&nbsp; &nbsp; &nbsp; &nbsp; Train_X</span></span><span style="color:black">, Test_X, Train_Y, Test_Y = model_selection.train_test_split(txt_text,txt_label)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; count_vect = CountVectorizer(analyzer=</span><em><span style="color:#00aa00">&#39;word&#39;</span></em><span style="color:black">)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; count_vect.fit(txt_text)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:silver"># transform the training and validation data using count <u>vectorizer</u> object</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Train_X_count =&nbsp; count_vect.transform(<span style="background-color:#ffff96">Train_X</span>)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Test_X_count =&nbsp; count_vect.transform(Test_X)&nbsp;&nbsp; </span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><strong><u><span style="color:#333333">TF-IDF Vectors as features</span></u></strong></span></span></span></p>

<p style="margin-left:0in; margin-right:0in; text-align:justify"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><span style="color:#595858">TF-IDF score represents the relative importance of a term in the document and the entire corpus. TF-IDF score is composed by two terms: the first computes the normalized Term Frequency (TF), the second term is the Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents in the corpus divided by the number of documents where the specific term appears.</span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in; text-align:justify"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><span style="color:#595858">TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)<br />
IDF(t) = log_e(Total number of documents / Number of documents with term t in it)</span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in; text-align:justify"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><span style="color:#595858">TF-IDF Vectors can be generated at different levels of input tokens (words, characters, n-grams)</span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in; text-align:justify"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><strong><span style="color:#333333">a. Word Level TF-IDF :</span></strong><span style="color:#595858">&nbsp;Matrix representing tf-idf scores of every term in different documents</span><br />
<strong><span style="color:#333333">b. N-gram Level TF-IDF :</span></strong><span style="color:#595858">&nbsp;N-grams are the combination of N terms together. This Matrix representing tf-idf scores of N-grams</span><br />
<strong><span style="color:#333333">c. Character Level TF-IDF :</span></strong><span style="color:#595858">&nbsp;Matrix representing tf-idf scores of character level n-grams in the corpus</span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><strong><u><span style="color:#333333">Word Level TF-IDF :</span>&nbsp;</u></strong></span></span></p>

<p style="margin-left:0in; margin-right:0in">&nbsp;</p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp; &nbsp; &nbsp; &nbsp;Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(txt_text, txt_label)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;<span style="color:silver"># split the <u>dataset</u> into training and validation <u>datasets</u> </span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span style="color:silver"># label encode the target variable </span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; encoder = preprocessing.LabelEncoder()</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Train_Y = encoder.fit_transform(Train_Y)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Test_Y = encoder.fit_transform(Test_Y)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:silver"># split the <u>dataset</u> into training and validation <u>datasets</u> </span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; tfidf_vect = TfidfVectorizer(analyzer=</span><em><span style="color:#00aa00">&#39;word&#39;</span></em><span style="color:black">,max_features=</span><span style="color:maroon">5000</span><span style="color:black">)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; tfidf_vect.fit(txt_text)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Train_X_Tfidf = tfidf_vect.transform(Train_X)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Test_X_Tfidf = tfidf_vect.transform(Test_X)</span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><span style="color:black">This code segment fond in </span><span style="background-color:#ffff96"><span style="color:black">word_tf_idf.py</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><strong><u><span style="color:#333333">N-gram Level TF-IDF:</span></u></strong></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:#ffff96"><span style="color:black">&nbsp; &nbsp; &nbsp; &nbsp; Train_X</span></span><span style="color:black">, Test_X, Train_Y, Test_Y = model_selection.train_test_split(txt_text, txt_label)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp; &nbsp; &nbsp; &nbsp;<span style="color:black">tfidf_vect_ngram = TfidfVectorizer(analyzer=</span><em><span style="color:#00aa00">&#39;word&#39;</span></em><span style="color:black">, ngram_range=(</span><span style="color:maroon">2</span><span style="color:black">, </span><span style="color:maroon">3</span><span style="color:black">), max_features=</span><span style="color:maroon">5000</span><span style="color:black">)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; tfidf_vect_ngram.fit(txt_text)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Train_X_ngram = tfidf_vect_ngram.transform(<span style="background-color:#ffff96">Train_X</span>)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Test_X_ngram = tfidf_vect_ngram.transform(Test_X)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><strong><span style="color:black">This code segment fond in </span><span style="background-color:#ffff96"><span style="color:black">ngram_tf_idf.py</span></span></strong></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><strong><strong><u><span style="color:#333333">Character Level TF-IDF:</span></u></strong></strong></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp; &nbsp; &nbsp; &nbsp; Train_X, Test_X, Train_Y, Test_Y&nbsp; = model_selection.train_test_split(txt_text, txt_label)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; tfidf_vect_ngram_chars = TfidfVectorizer(analyzer=</span><em><span style="color:#00aa00">&#39;char&#39;</span></em><span style="color:black">, token_pattern=</span><em><span style="color:#00aa00">r&#39;\w{1,}&#39;</span></em><span style="color:black">, ngram_range=(</span><span style="color:maroon">2</span><span style="color:black">,</span><span style="color:maroon">3</span><span style="color:black">), max_features=</span><span style="color:maroon">5000</span><span style="color:black">)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; tfidf_vect_ngram_chars.fit(txt_text)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Train_X_ngram_chars =&nbsp; tfidf_vect_ngram_chars.transform(Train_X) </span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Test_X_ngram_chars =&nbsp; tfidf_vect_ngram_chars.transform(Test_X)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><strong><u><span style="background-color:white"><span style="color:#333333">Model Training &amp; evaluate the performance of model:</span></span></u></strong></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><span style="color:#595858">The final step in the text classification framework is to train a classifier using the features created in the previous step. There are many different choices of machine learning models which can be used to train a final model. We will implement following different classifiers for this purpose:</span></span></span></span></p>

<ol>
	<li>
	<p><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><span style="color:#595858">Naive Bayes Classifier</span></span></span></span></p>
	</li>
	<li>
	<p><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><span style="color:#595858">Linear Classifier</span></span></span></span></p>
	</li>
	<li>
	<p><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><span style="color:#595858">Support Vector Machine</span></span></span></span></p>
	</li>
	<li>
	<p><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><span style="color:#595858">Bagging Models</span></span></span></span></p>
	</li>
</ol>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><span style="color:#595858">Lets implement these models and understand their details. The following function is a utility function which can be used to train a model. It accepts the classifier, feature_vector of training data, labels of training data and feature vectors of valid data as inputs. Using these inputs, the model is trained and accuracy score is computed.</span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:blue">&nbsp; &nbsp; &nbsp; def</span><span style="color:black"> <strong>train_model</strong>(<em>self</em>,classifier, train_input,test_input, train_target, test_target, is_neural_net=</span><span style="color:blue">False</span><span style="color:black">):</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;<span style="color:silver"># fit the training <u>dataset</u> on the classifie</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; classifier.fit(train_input, train_target)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span style="color:silver"># predict the labels on validation <u>dataset</u></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; predictions = classifier.predict(test_input)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;<span style="color:blue">if</span><span style="color:black"> is_neural_net:</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; predictions = predictions.argmax(axis=-</span><span style="color:maroon">1</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span style="color:silver">#print(classifier.predict(gettfidf_vect.transform([&quot;A FIVE STAR BOOK&quot;])))</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span style="color:blue">return</span><span style="color:black"> accuracy_score(predictions, test_target)</span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><u><span style="color:#333333">Naive Bayes:</span></u></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><span style="color:#595858">Implementing a naive bayes model using sklearn implementation with different features</span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><span style="color:#595858">Naive Bayes is a classification technique based on Bayes&rsquo; Theorem with an assumption of independence among predictors. A Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature</span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in">&nbsp;</p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:silver">&nbsp; &nbsp; &nbsp; &nbsp;# Text feature engineering with char_tf_idf </span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp; &nbsp; &nbsp; &nbsp;model_input = char_tf_idf().convert_feature(txt_text, txt_label)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span style="color:silver"># Text feature engineering</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span style="color:silver">#&nbsp; Build Text Classification Model and Evaluating the Model</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp; &nbsp; &nbsp; &nbsp;naive = naive_bayes.MultinomialNB()</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp; &nbsp; &nbsp; &nbsp;accuracy = Classifier().train_model(naive, model_input.get_train_input(), model_input.get_test_input(),&nbsp;</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp; &nbsp; &nbsp; &nbsp;model_input.get_train_target(), </span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp; &nbsp; &nbsp; &nbsp;model_input.get_test_target())</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span style="color:blue">print</span><span style="color:black"> (</span><em><span style="color:#00aa00">&quot;NB, char_tf_idf accuracy is : &quot;</span></em><span style="color:black">, accuracy * </span><span style="color:maroon">100</span><span style="color:black">)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; </span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span style="color:silver"># Text feature engineering with count_vectorizer</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp; &nbsp; &nbsp; model_input = count_vectorizer().convert_feature(txt_text, txt_label)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span style="color:silver"># Text feature engineering with count_vectorizer</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; </span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;<span style="color:silver">#&nbsp; Build Text Classification Model and Evaluating the Model</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp; &nbsp; &nbsp;naive = naive_bayes.MultinomialNB()</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp; &nbsp; &nbsp;accuracy = Classifier().train_model(naive, model_input.get_train_input(), model_input.get_test_input(),&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;model_input.get_train_target(), model_input.get_test_target())</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;<span style="color:blue">print</span><span style="color:black"> (</span><em><span style="color:#00aa00">&quot;NB, count_vectorizer accuracy is : &quot;</span></em><span style="color:black">, accuracy * </span><span style="color:maroon">100</span><span style="color:black">)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;<span style="color:silver">#&nbsp; Build Text Classification Model and Evaluating the Model</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; </span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;<span style="color:silver"># Text feature engineering with ngram_tf_idf</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp; &nbsp; model_input = ngram_tf_idf().convert_feature(txt_text, txt_label)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp; &nbsp; &nbsp; &nbsp;<span style="color:silver"># Text feature engineering with ngram_tf_idf</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; </span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp; &nbsp; &nbsp; &nbsp;<span style="color:silver">#&nbsp; Build Text Classification Model and Evaluating the Model</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp; naive = naive_bayes.MultinomialNB()</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp; accuracy = Classifier().train_model(naive, model_input.get_train_input(), model_input.get_test_input(), model_input.get_train_target(), model_input.get_test_target())</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp; &nbsp; &nbsp; &nbsp;<span style="color:blue">print</span><span style="color:black"> (</span><em><span style="color:#00aa00">&quot;NB, ngram_tf_idf accuracy is : &quot;</span></em><span style="color:black">, accuracy * </span><span style="color:maroon">100</span><span style="color:black">)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; <span style="color:silver">#&nbsp; Build Text Classification Model and Evaluating the Model</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; </span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; <span style="color:silver"># Text feature engineering with word_tf_idf</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp; model_input = word_tf_idf().convert_feature(txt_text, txt_label)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; <span style="color:silver"># Text feature engineering with word_tf_idf</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; </span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; <span style="color:silver">#&nbsp; Build Text Classification Model and Evaluating the Model</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp; naive = naive_bayes.MultinomialNB()</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp; accuracy = Classifier().train_model(naive, model_input.get_train_input(), model_input.get_test_input(), model_input.get_train_target(), model_input.get_test_target())</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; <span style="color:blue">print</span><span style="color:black"> (</span><em><span style="color:#00aa00">&quot;NB, word_tf_idf accuracy is : &quot;</span></em><span style="color:black">, accuracy * </span><span style="color:maroon">100</span><span style="color:black">)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white">&nbsp;&nbsp;&nbsp; <span style="color:silver">#&nbsp; Build Text Classification Model and Evaluating the Model</span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">this code segment found in <strong><span style="background-color:#ffff96">Naive_Bay_Clf.py</span></strong></span></span></span></p>

<p style="margin-left:0in; margin-right:0in">&nbsp;</p>

<p style="margin-left:0in; margin-right:0in">&nbsp;</p>

<p style="margin-left:0in; margin-right:0in">&nbsp;</p>

<p style="margin-left:0in; margin-right:0in">&nbsp;</p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">NB, char_tf_idf accuracy is :&nbsp; 81.28</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">NB, count_vectorizer accuracy is :&nbsp; 82.96</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">NB, ngram_tf_idf accuracy is :&nbsp; 81.92</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><span style="color:black">NB, word_tf_idf accuracy is :&nbsp; 85.96000000000001</span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in">&nbsp;</p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><strong><u><span style="color:#595858">Linear Classifier</span></u></strong></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><span style="color:#595858">Implementing a Linear Classifier (Logistic Regression)</span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><span style="color:#595858">Logistic regression measures the relationship between the categorical dependent variable and one or more independent variables by estimating probabilities using a logistic/sigmoid function. </span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:silver">#&nbsp; Build Text Classification Model and Evaluating the Model</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp; naive = linear_model.LogisticRegression()</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp; accuracy = Classifier().train_model(naive, model_input.get_train_input(), model_input.get_test_input(), model_input.get_train_target(), model_input.get_test_target())</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; <span style="color:blue">print</span><span style="color:black"> (</span><em><span style="color:#00aa00">&quot;Linear_Clf, count_vectorizer accuracy is : &quot;</span></em><span style="color:black">, accuracy * </span><span style="color:maroon">100</span><span style="color:black">)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; </span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; <span style="color:silver"># Text feature engineering </span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp; model_input = ngram_tf_idf().convert_feature(txt_text, txt_label)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; <span style="color:silver"># Text feature engineering </span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; </span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; <span style="color:silver">#&nbsp; Build Text Classification Model and Evaluating the Model</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp; naive = linear_model.LogisticRegression()</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp; accuracy = Classifier().train_model(naive, model_input.get_train_input(), model_input.get_test_input(), model_input.get_train_target(), model_input.get_test_target())</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; <span style="color:blue">print</span><span style="color:black"> (</span><em><span style="color:#00aa00">&quot;Linear_Clf, ngram_tf_idf accuracy is : &quot;</span></em><span style="color:black">, accuracy * </span><span style="color:maroon">100</span><span style="color:black">)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; </span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; <span style="color:silver"># Text feature engineering </span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp; model_input = word_tf_idf().convert_feature(txt_text, txt_label)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; <span style="color:silver"># Text feature engineering </span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; </span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; <span style="color:silver">#&nbsp; Build Text Classification Model and Evaluating the Model</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp; naive = linear_model.LogisticRegression()</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp; accuracy = Classifier().train_model(naive, model_input.get_train_input(), model_input.get_test_input(), model_input.get_train_target(), model_input.get_test_target())</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white">&nbsp;&nbsp;&nbsp; <span style="color:blue">print</span><span style="color:black"> (</span><em><span style="color:#00aa00">&quot;Linear_Clf, word_tf_idf accuracy is : &quot;</span></em><span style="color:black">, accuracy * </span><span style="color:maroon">100</span><span style="color:black">)</span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in">&nbsp;</p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">Linear_Clf, char_tf_idf accuracy is :&nbsp; 84.36</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">Linear_Clf, count_vectorizer accuracy is :&nbsp; 85.92</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">Linear_Clf, ngram_tf_idf accuracy is :&nbsp; 82.64</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">Linear_Clf, word_tf_idf accuracy is :&nbsp; 87.4</span></span></span></p>

<p style="margin-left:0in; margin-right:0in">&nbsp;</p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><strong>&nbsp;</strong><u><span style="color:#333333">SVM Model</span></u></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><span style="background-color:white"><span style="color:#595858">Support Vector Machine (SVM) is a supervised machine learning algorithm which can be used for both classification or regression challenges. The model extracts a best possible hyper-plane / line that segregates the two classes.</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:silver">#&nbsp; Build Text Classification Model and Evaluating the Model</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp; naive=svm.SVC()</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp; accuracy = Classifier().train_model(naive,model_input.get_train_input(),model_input.get_test_input(), model_input.get_train_target(), model_input.get_test_target())</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white">&nbsp;&nbsp;&nbsp; <span style="color:blue">print</span><span style="color:black"> (</span><em><span style="color:#00aa00">&quot;Svm_clf, ngram_tf_idf accuracy is : &quot;</span></em><span style="color:black">, accuracy*</span><span style="color:maroon">100</span><span style="color:black">)</span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in">&nbsp;</p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><span style="color:black">Svm_clf, ngram_tf_idf accuracy is :&nbsp; 51.76</span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in">&nbsp;</p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><u><span style="background-color:white"><span style="color:#595858">Random Forest Model</span></span></u></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><span style="background-color:white"><span style="color:#595858">Random Forest models are a type of ensemble models, particularly bagging models. They are part of the tree based model family.</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:silver">#&nbsp; Build Text Classification Model and Evaluating the Model</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp; naive=ensemble.RandomForestClassifier()</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp; accuracy = Classifier().train_model(naive,model_input.get_train_input(),model_input.get_test_input(), model_input.get_train_target(), model_input.get_test_target())</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; <span style="color:blue">print</span><span style="color:black"> (</span><em><span style="color:#00aa00">&quot;RandomForest_Clf, count_vectorizer accuracy is : &quot;</span></em><span style="color:black">, accuracy*</span><span style="color:maroon">100</span><span style="color:black">)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; </span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; </span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; <span style="color:silver">#Text feature engineering </span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp; model_input=word_tf_idf().convert_feature(clear_txt,txt_label)</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; <span style="color:silver">#Text feature engineering </span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; </span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif">&nbsp;&nbsp;&nbsp; <span style="color:silver">#&nbsp; Build Text Classification Model and Evaluating the Model</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp; naive=ensemble.RandomForestClassifier()</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">&nbsp;&nbsp;&nbsp; accuracy = Classifier().train_model(naive,model_input.get_train_input(),model_input.get_test_input(), model_input.get_train_target(), model_input.get_test_target())</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white">&nbsp;&nbsp;&nbsp; <span style="color:blue">print</span><span style="color:black"> (</span><em><span style="color:#00aa00">&quot;RandomForest_Clf, word_tf_idf accuracy is : &quot;</span></em><span style="color:black">, accuracy*</span><span style="color:maroon">100</span><span style="color:black">)</span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in">&nbsp;</p>

<p style="margin-left:0in; margin-right:0in">&nbsp;</p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">RandomForest_Clf, count_vectorizer accuracy is :&nbsp; 77.84</span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="color:black">RandomForest_Clf, word_tf_idf accuracy is :&nbsp; 78.52</span></span></span></p>

<p style="margin-left:0in; margin-right:0in">&nbsp;</p>
