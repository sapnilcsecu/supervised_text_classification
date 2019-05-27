<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><strong><u>load dataset</u></strong></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:12.0pt"><span style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;"><span style="color:#595858">To prepare the dataset, load the downloaded data into a pandas dataframe</span></span></span><span style="font-size:12.0pt"><span style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;"> <span style="color:#595858">containing two columns &ndash; text and label.</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in">&nbsp;</p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">&nbsp; load_data = dataset_load();</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">&nbsp; trainDF = load_data.load_cvs_dataset(</span></span></span><em><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:#00aa00">&quot;../corpus.csv&quot;</span></span></span></em><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">)</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">&nbsp; txt_label = trainDF[</span></span></span><em><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:#00aa00">&#39;label&#39;</span></span></span></em><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">]</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">&nbsp; txt_text = trainDF[</span></span></span><em><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:#00aa00">&#39;text&#39;</span></span></span></em><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">]</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in">&nbsp;</p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif">this code segment&nbsp; found in trainmodel_write.py</span></span></p>

<p style="margin-left:0in; margin-right:0in">&nbsp;</p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:blue">&nbsp; def</span></span></span><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black"> <strong>load_cvs_dataset</strong>(<em>self</em>,dataset_path):</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:silver">&nbsp; &nbsp; &nbsp;#Set Random seed</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">&nbsp; &nbsp; &nbsp;np.random.seed(</span></span></span><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:maroon">500</span></span></span><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">)</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:silver">&nbsp; &nbsp; &nbsp;# Add the Data using pandas</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">&nbsp; &nbsp; &nbsp;Corpus = pd.read_csv(dataset_path,encoding=</span></span></span><em><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:#00aa00">&#39;<u>latin</u>-1&#39;</span></span></span></em><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">,error_bad_lines=</span></span></span><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:blue">False</span></span></span><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">)</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:blue">&nbsp; return</span></span></span><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black"> Corpus</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">this code segment found in <strong><span style="background-color:#ffff96">dataset_load.py</span></strong></span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in">&nbsp;</p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><strong><u>Text Feature Engineering:</u></strong></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-family:Times New Roman,Times,serif"><span style="font-size:11pt"><span style="background-color:white"><span style="font-size:11.5pt"><span style="color:#595858">The next step is the feature engineering step. In this step, raw text data will be transformed into feature vectors and new features will be created using the existing dataset. We will implement the following different ideas in order to obtain relevant features from our dataset.</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-family:Times New Roman,Times,serif"><span style="font-size:11pt"><span style="background-color:white"><span style="font-size:11.5pt"><span style="color:#595858">Count Vectors as features<br />
TF-IDF Vectors as features</span></span></span></span></span></p>

<ul>
	<li><span style="font-family:Times New Roman,Times,serif"><span style="font-size:11pt"><span style="background-color:white"><span style="color:#595858"><span style="font-size:11.5pt">Word level</span></span></span></span></span></li>
	<li><span style="font-family:Times New Roman,Times,serif"><span style="font-size:11pt"><span style="background-color:white"><span style="color:#595858"><span style="font-size:11.5pt">N-Gram level</span></span></span></span></span></li>
	<li><span style="font-family:Times New Roman,Times,serif"><span style="font-size:11pt"><span style="background-color:white"><span style="color:#595858"><span style="font-size:11.5pt">Character level</span></span></span></span></span></li>
</ul>

<p style="margin-left:0in; margin-right:0in"><span style="font-family:Times New Roman,Times,serif"><span style="font-size:11pt"><span style="background-color:white"><span style="font-size:11.5pt"><span style="color:#595858">Lets look at the implementation of these ideas in detail.</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:12px"><span style="font-family:Times New Roman,Times,serif"><span style="background-color:white"><strong><u><span style="color:#333333">Count Vectors as features</span></u></strong></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-family:Times New Roman,Times,serif"><span style="font-size:11pt"><span style="font-size:11.5pt"><span style="background-color:white"><span style="color:#595858">Count Vector is a matrix notation of the dataset in which every row represents a document from the corpus, every column represents a term from the corpus, and every cell represents the frequency count of a particular term in a particular document.</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in">&nbsp;</p>

<p style="margin-left:0in; margin-right:0in"><span style="font-family:Times New Roman,Times,serif"><span style="font-size:11pt"><span style="font-size:10.0pt"><span style="background-color:#ffff96"><span style="color:black">&nbsp; &nbsp; &nbsp; &nbsp; Train_X</span></span><span style="color:black">, Test_X, Train_Y, Test_Y = model_selection.train_test_split(txt_text,txt_label)</span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-family:Times New Roman,Times,serif"><span style="font-size:11pt"><span style="font-size:10.0pt"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; count_vect = CountVectorizer(analyzer=</span></span><em><span style="font-size:10.0pt"><span style="color:#00aa00">&#39;word&#39;</span></span></em><span style="font-size:10.0pt"><span style="color:black">)</span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-family:Times New Roman,Times,serif"><span style="font-size:11pt"><span style="font-size:10.0pt"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; count_vect.fit(txt_text)</span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-family:Times New Roman,Times,serif"><span style="font-size:11pt">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-family:Times New Roman,Times,serif"><span style="font-size:11pt">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="font-size:10.0pt"><span style="color:silver"># transform the training and validation data using count <u>vectorizer</u> object</span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-family:Times New Roman,Times,serif"><span style="font-size:11pt"><span style="font-size:10.0pt"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Train_X_count =&nbsp; count_vect.transform(<span style="background-color:#ffff96">Train_X</span>)</span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-family:Times New Roman,Times,serif"><span style="font-size:11pt"><span style="font-size:10.0pt"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Test_X_count =&nbsp; count_vect.transform(Test_X)&nbsp;&nbsp; </span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-family:Times New Roman,Times,serif"><span style="font-size:12px"><span style="background-color:white"><strong><u><span style="color:#333333">TF-IDF Vectors as features</span></u></strong></span></span></span></p>

<p style="margin-left:0in; margin-right:0in; text-align:justify"><span style="font-family:Times New Roman,Times,serif"><span style="font-size:12pt"><span style="background-color:white"><span style="font-size:11.5pt"><span style="color:#595858">TF-IDF score represents the relative importance of a term in the document and the entire corpus. TF-IDF score is composed by two terms: the first computes the normalized Term Frequency (TF), the second term is the Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents in the corpus divided by the number of documents where the specific term appears.</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in; text-align:justify"><span style="font-family:Times New Roman,Times,serif"><span style="font-size:12pt"><span style="background-color:white"><span style="font-size:11.5pt"><span style="color:#595858">TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)<br />
IDF(t) = log_e(Total number of documents / Number of documents with term t in it)</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in; text-align:justify"><span style="font-family:Times New Roman,Times,serif"><span style="font-size:12pt"><span style="background-color:white"><span style="font-size:11.5pt"><span style="color:#595858">TF-IDF Vectors can be generated at different levels of input tokens (words, characters, n-grams)</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in; text-align:justify"><span style="font-family:Times New Roman,Times,serif"><span style="font-size:12pt"><span style="background-color:white"><strong><span style="font-size:11.5pt"><span style="color:#333333">a. Word Level TF-IDF :</span></span></strong><span style="font-size:11.5pt"><span style="color:#595858">&nbsp;Matrix representing tf-idf scores of every term in different documents</span></span><br />
<strong><span style="font-size:11.5pt"><span style="color:#333333">b. N-gram Level TF-IDF :</span></span></strong><span style="font-size:11.5pt"><span style="color:#595858">&nbsp;N-grams are the combination of N terms together. This Matrix representing tf-idf scores of N-grams</span></span><br />
<strong><span style="font-size:11.5pt"><span style="color:#333333">c. Character Level TF-IDF :</span></span></strong><span style="font-size:11.5pt"><span style="color:#595858">&nbsp;Matrix representing tf-idf scores of character level n-grams in the corpus</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><strong><u><span style="font-size:12.0pt"><span style="color:#333333">Word Level TF-IDF :</span></span></u></strong><strong><u>&nbsp;</u></strong></span></span></p>

<p style="margin-left:0in; margin-right:0in">&nbsp;</p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">&nbsp; &nbsp; &nbsp; &nbsp;Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(txt_text, txt_label)</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;<span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:silver"># split the <u>dataset</u> into training and validation <u>datasets</u> </span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:silver"># label encode the target variable </span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; encoder = preprocessing.LabelEncoder()</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Train_Y = encoder.fit_transform(Train_Y)</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Test_Y = encoder.fit_transform(Test_Y)</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:silver"># split the <u>dataset</u> into training and validation <u>datasets</u> </span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; tfidf_vect = TfidfVectorizer(analyzer=</span></span></span><em><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:#00aa00">&#39;word&#39;</span></span></span></em><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">,max_features=</span></span></span><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:maroon">5000</span></span></span><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">)</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; tfidf_vect.fit(txt_text)</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Train_X_Tfidf = tfidf_vect.transform(Train_X)</span></span></span></span></span></p>

<h3 style="margin-left:0in; margin-right:0in"><span style="font-size:13.5pt"><span style="background-color:white"><span style="font-family:&quot;Times New Roman&quot;,serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Test_X_Tfidf = tfidf_vect.transform(Test_X)</span></span></span></span></span></span></h3>

<h3 style="margin-left:0in; margin-right:0in"><span style="font-size:13.5pt"><span style="background-color:white"><span style="font-family:&quot;Times New Roman&quot;,serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">This code segment fond in </span></span></span><span style="font-size:10.0pt"><span style="background-color:#ffff96"><span style="font-family:Consolas"><span style="color:black">word_tf_idf.py</span></span></span></span></span></span></span></h3>


