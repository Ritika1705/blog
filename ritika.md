---


---

<h2 id="machine-learning-for-noobs">MACHINE LEARNING FOR NOOBS</h2>
<p>Sir Arthur C. Clarke once said “ any sufficiently advanced technology is equivalent to magic”.<br>
Machine learning in my opinion would be one such magic.<br>
Have you seen how a small kid learns by repeating the same mistake again and again? That is exactly the entire concept of machine learning. Machine learning focuses on development of computer programs that can access data and use it for themselves.<br>
From self driving cars to Elon Musk trying to hook up brains directly to computers, the world stands totally unpredictable with the enormous growth in the field of machine learning.<br>
Machine learning can be broadly classified into two main categories:</p>
<ol>
<li>Supervised learning</li>
<li>Unsupervised learning</li>
</ol>
<p><strong>SUPERVISED MACHINE LEARNING:</strong></p>
<p><em>Supervised learning is the  task of learning a function that maps an input to an output based on example input-output pairs. It infers from a function  consisting of a set of <em>training examples</em>. In supervised learning, each example is a <em>pair</em> consisting of an input object (typically a vector) and a desired output value (also called the <em>supervisory signal</em>). A supervised learning algorithm analyzes the training data and produces an inferred function, which can be used for mapping new examples.</em></p>
<p><strong>UNSUPERVISED MACHINE LEARNING</strong></p>
<p><em>Unsupervised learning is a type of machine learning based algorithm where the outputs cannot be predicted based on previous results. It can’t be applied to regression or classification problem.<br>
It is used to discover the previously known patterns or trends in the data and predict the desired results.<br>
Now suppose we are given with the prices of various appliances over the years and are told to find the hidden trends in the given data. Here , the price of an appliance is totally dependent on the quality and productivity of that particular appliance and hence the price.<br>
This is the entire concept of a simple regression problem in machine learning.<br>
Linear regression establishes a relationship between two different variables by fitting them into a linear equation. one of the variable is considered to be independent whereas the other is dependent.<br>
So let’s get started!!</em></p>
<p><strong>STEP 1: IMPORTING LIBRARIES</strong><br>
<em>As stated earlier, machine learning is the science where computer learns from different huge data sets. Hence it becomes quite a tedious tasks to perform these operations manually using codes and mathematical statistical formula. Hence to make it more efficient and easy, there are various python libraries, frameworks and modules:</em></p>
<p>import pandas as pd<br>
import numpy as np<br>
import matplotlib.pyplot as plt from sklearn.linear_model import LinearRegression</p>
<p><strong>STEP 2: READING THE DATA</strong></p>
<p><em>The first step after downloading the data set is reading the it into the code. CSV stands for comma separated value. These files are a common format for storing as well as transferring huge amounts of data and importing these files is a key step for training any kind of model in machine learning.</em></p>
<p>data = pd.read_csv(“data/Nameofthefile.csv”)</p>
<p><strong>STEP 3: DATA MODELLING</strong></p>
<p><em>Data modelling is one of the most interesting part of any machine learning. It basically refers to the process of feeding the machine with data instructions and visualizing how are model will actually look like.</em></p>
<p>plt.figure(figsize=(16, 8))<br>
plt.scatter(<br>
data[‘attribute1’],<br>
data[‘attribute2’],<br>
c=‘black’<br>
)</p>
<p>plt.xlabel(“………… (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mo stretchy="false">)</mo><mi mathvariant="normal">"</mi><mo stretchy="false">)</mo><mi>p</mi><mi>l</mi><mi>t</mi><mi mathvariant="normal">.</mi><mi>y</mi><mi>l</mi><mi>a</mi><mi>b</mi><mi>e</mi><mi>l</mi><mo stretchy="false">(</mo><mi mathvariant="normal">"</mi><mo>…</mo><mo>…</mo><mo>…</mo><mo stretchy="false">(</mo></mrow><annotation encoding="application/x-tex">)")
plt.ylabel("……… (</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1em; vertical-align: -0.25em;"></span><span class="mclose">)</span><span class="mord">"</span><span class="mclose">)</span><span class="mord mathdefault">p</span><span class="mord mathdefault" style="margin-right: 0.01968em;">l</span><span class="mord mathdefault">t</span><span class="mord">.</span><span class="mord mathdefault" style="margin-right: 0.03588em;">y</span><span class="mord mathdefault" style="margin-right: 0.01968em;">l</span><span class="mord mathdefault">a</span><span class="mord mathdefault">b</span><span class="mord mathdefault">e</span><span class="mord mathdefault" style="margin-right: 0.01968em;">l</span><span class="mopen">(</span><span class="mord">"</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="minner">…</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="minner">…</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="minner">…</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mopen">(</span></span></span></span></span>)")<br>
plt.show()<br>
<img src="https://lh3.googleusercontent.com/BaKz-9PBwYgESDMi-loKYxuD_VGKZ33xvSXFgl5pSypt0Q0240JpPs3RR40N0Lvdsi_AC1TAXJw" alt=""></p>
<p><strong>STEP 4: GENERALIZATION:</strong></p>
<p><em>Have you ever wondered why examinations are conducted in any educational institution be it big or small???<br>
It is to test how well you have mastered the concepts and are ready to implement those. This is what is referred to as generalization.<br>
It refers to how well the concepts learned by a machine learning model apply to specific examples.</em></p>
<p>X = data[‘attribute’].values.reshape(-1,1)<br>
y = data[‘attribute2’].values.reshape(-1,1)<br>
reg = LinearRegression()<br>
reg.fit(X, y)</p>
<p>print(“The linear model is: Y = {:.5} + {:.5}X”.format(reg.intercep_[0], reg.coef_[0][0]))<br>
predictions = reg.predict(X)<br>
plt.figure(figsize=(16, 8))<br>
plt.scatter(<br>
data[‘attribute1’],<br>
data[‘attribute2’],<br>
c=‘black’<br>
)<br>
plt.plot(<br>
data[‘attribute1’],<br>
predictions,<br>
c=‘blue’,<br>
linewidth=2<br>
)<br>
plt.xlabel("……………………….")<br>
plt.ylabel("………………………")<br>
plt.show()<br>
<img src="https://lh3.googleusercontent.com/U8S5_ac1TTp5rJE50bepe8BhIGxn46JVQVQQyx1tyjBe1DCzDbMcBV1bANbMBv9K7BJfd7Nm6xA" alt="enter image description here"></p>
<p>WOW!!!</p>
<p>Isn’t  it great??</p>
<p>Hence machine learning is nothing but statistics. From evaluating millions and millions and millions of data, it has become an integral part of almost all sectors of the world, be it big or small.</p>
<p>From government to healthcare  to transportation it has become an indelible part of our fast pacing lives.<br>
Nick Bostrom once said " Machine Intelligence is the last invention that humanity will ever need to make".<br>
And hence we stand today with the greatest invention of the era and making the world more powerful and equipped than it used to be.</p>

