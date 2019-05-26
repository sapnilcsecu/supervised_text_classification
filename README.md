<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><strong><u><span style="font-size:12.0pt"><span style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">load dataset</span></span></u></strong></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:12.0pt"><span style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;"><span style="color:#595858">To prepare the dataset, load the downloaded data into a pandas dataframe</span></span></span><span style="font-size:12.0pt"><span style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;"> <span style="color:#595858">containing two columns &ndash; text and label.</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">load_data = dataset_load();</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">trainDF = load_data.load_cvs_dataset(</span></span></span><em><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:#00aa00">&quot;../corpus.csv&quot;</span></span></span></em><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">)</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">txt_label = trainDF[</span></span></span><em><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:#00aa00">&#39;label&#39;</span></span></span></em><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">]</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">txt_text = trainDF[</span></span></span><em><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:#00aa00">&#39;text&#39;</span></span></span></em><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">]</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif">this code segment&nbsp; found in <a href="https://github.com/sapnilcsecu/supervised_text_classification/blob/master/super_text_class/example/trainmodel_write.py">trainmodel_write.py</a></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:blue">def</span></span></span><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black"> <strong>load_cvs_dataset</strong>(<em>self</em>,dataset_path):</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:silver">#Set Random seed</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">np.random.seed(</span></span></span><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:maroon">500</span></span></span><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">)</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:silver"># Add the Data using pandas</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">Corpus = pd.read_csv(dataset_path,encoding=</span></span></span><em><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:#00aa00">&#39;<u>latin</u>-1&#39;</span></span></span></em><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">,error_bad_lines=</span></span></span><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:blue">False</span></span></span><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">)</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:blue">return</span></span></span><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black"> Corpus</span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:10.0pt"><span style="font-family:Consolas"><span style="color:black">this code segment found in <a href="https://github.com/sapnilcsecu/supervised_text_classification/blob/master/super_text_class/dataset_pre/dataset_load.py"><strong><span style="background-color:#ffff96">dataset_load.py</span></strong></a></span></span></span></span></span></p>



<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><strong><u><span style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">Text Feature Engineering:</span></u></strong></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="background-color:white"><span style="font-family:Calibri,sans-serif"><span style="font-size:11.5pt"><span style="font-family:&quot;Arial&quot;,&quot;sans-serif&quot;"><span style="color:#595858">The next step is the feature engineering step. In this step, raw text data will be transformed into feature vectors and new features will be created using the existing dataset. We will implement the following different ideas in order to obtain relevant features from our dataset.</span></span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="background-color:white"><span style="font-family:Calibri,sans-serif"><span style="font-size:11.5pt"><span style="font-family:&quot;Arial&quot;,&quot;sans-serif&quot;"><span style="color:#595858">Count Vectors as features<br />
TF-IDF Vectors as features</span></span></span></span></span></span></p>

<ul>
	<li><span style="font-size:11pt"><span style="background-color:white"><span style="color:#595858"><span style="font-family:Calibri,sans-serif"><span style="font-size:11.5pt"><span style="font-family:&quot;Arial&quot;,&quot;sans-serif&quot;">Word level</span></span></span></span></span></span></li>
	<li><span style="font-size:11pt"><span style="background-color:white"><span style="color:#595858"><span style="font-family:Calibri,sans-serif"><span style="font-size:11.5pt"><span style="font-family:&quot;Arial&quot;,&quot;sans-serif&quot;">N-Gram level</span></span></span></span></span></span></li>
	<li><span style="font-size:11pt"><span style="background-color:white"><span style="color:#595858"><span style="font-family:Calibri,sans-serif"><span style="font-size:11.5pt"><span style="font-family:&quot;Arial&quot;,&quot;sans-serif&quot;">Character level</span></span></span></span></span></span></li>
</ul>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="background-color:white"><span style="font-family:Calibri,sans-serif"><span style="font-size:11.5pt"><span style="font-family:&quot;Arial&quot;,&quot;sans-serif&quot;"><span style="color:#595858">Lets look at the implementation of these ideas in detail.</span></span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="background-color:white"><span style="font-family:Calibri,sans-serif"><strong><u><span style="font-size:12.0pt"><span style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;"><span style="color:#333333">Count Vectors as features</span></span></span></u></strong></span></span></span></p>

<p style="margin-left:0in; margin-right:0in"><span style="font-size:11pt"><span style="font-family:Calibri,sans-serif"><span style="font-size:11.5pt"><span style="background-color:white"><span style="font-family:&quot;Arial&quot;,&quot;sans-serif&quot;"><span style="color:#595858">Count Vector is a matrix notation of the dataset in which every row represents a document from the corpus, every column represents a term from the corpus, and every cell represents the frequency count of a particular term in a particular document.</span></span></span></span></span></span></p>

<p style="margin-left:0in; margin-right:0in">&nbsp;</p>
