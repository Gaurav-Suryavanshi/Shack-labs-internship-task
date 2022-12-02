Completed the following 2 tasks as part of Shack Labs DS Internship Assignment: 

<ol>
  <li>House Price Prediction</li>
  <li>Matching of Amazon and Flipkart Products</li>
</ol>

## Libraries & Frameworks Used

<h3>For Task 1: House Price Prediction</h3>
<ul type='a'>
  <li>Numpy</li>
  <li>Pandas</li>
  <li>Matplotlib</li>
  <li>Seaborn</li>
  <li>Scikit-learn</li>
  <li>Scipy</li>
  <li>XGBoost</li>
  <li>LightGBM</li>
  <li>CatBoost</li>
</ul>

<h3>For Task 2: Product Matching</h3>
<ul type='a'>
  <li>PyTorch</li>
  <li>Sentence Transformers</li>
  <li>Pandas</li>
</ul>

## Machine Learning Models Performance Comparison for Task 1: House Price Prediction

**Drawbacks related to Underlying Assumptions of Machine Learning Algorithms**

The assumptions associated with regression modeling and machine learning in general are as follows:

<ul>
  <li>Multicollinearity : There is minimal or no multicollinearity among the independent variables. It usually requires a large sample size to predict properly. It assumes the observations to be independent of each other.</li>
  <li>Homoscedasticity: The variance of residual should be the same for any value of X.</li>
  <li>Observations are assumed to be independent of each other.</li>
  <li>For most of the machine learning algorithms such as Linear Regression and many clustering algorithms, normal distribution is necessary for producing better outcomes.</li>
  <li>Unstability: Tree-based models are relatively unstable. A small change in the data can cause a large change in the structure of the decision tree causing instability.</li>
  <li>Non-scalability: Several boosting models such as XGBoost and Gradient Boosting models are very sensitive to outliers since every classifier is forced to fix the errors in the predecessor learners. The overall method is hardly scalable. Moreover, they don't perform well enough on sparse and unstructured data due to their internal working and underlying assumptions.</li>
</ul>

<table>
  <tr>
    <th>Model</th>
    <th>R2 Score (%)</th>
  </tr>
  <tr>
    <td>Extra Trees Regressor</td>
    <td>81.64</td>
  </tr>
  <tr>
    <td>Cat Boost Regressor</td>
    <td>81.39</td>
  </tr>
  <tr>
    <td>Gradient Boosting Regressor</td>
    <td>78.96</td>
  </tr>
  <tr>
    <td>XG Boost Regressor</td>
    <td>72.87</td>
  </tr>
  <tr>
    <td>Random Forest Regressor</td>
    <td>75.49</td>
  </tr>
  <tr>
    <td>Decision Tree Regressor</td>
    <td>63.80</td>
  </tr>
  <tr>
    <td>K Nearest Neighbors Regressor</td>
    <td>57.61</td>
  </tr>
  <tr>
    <td>Support Vector Regressor</td>
    <td>59.28</td>
  </tr>
  <tr>
    <td>Light Gradient Boosting Regressor</td>
    <td>74.72</td>
  </tr>
  <tr>
    <td>Bagging Regressor</td>
    <td>74.82</td>
  </tr>
  <tr>
    <td>Linear Regression</td>
    <td>64.39</td>
  </tr>
  <tr>
    <td>Multi-Layer Perceptron Regressor</td>
    <td>51.15</td>
  </tr>
  <tr>
    <td>Histogram Gradient Boosting Regressor</td>
    <td>78.42</td>
  </tr>
</table>

## Techniques for Semantic Comparison of Amazon and Flipkart product names

<p>Appropriate techniques that can be adopted for performing semantic comparison of two text documents are as follows:</p>

<ul>
  <li>Jaccard Index: Jaccard index, also known as Jaccard similarity coefficient,  treats the data objects like sets. It is defined as the size of the intersection of two sets divided by the size of the union.</li>
  <li>Euclidean Distance: Euclidean distance, or L2 norm, is the most commonly used form of the Minkowski distance. It uses the Pythagoras theorem to calculate the distance between two points.</li>
  <li>Cosine Similarity: Cosine Similarity computes the similarity of two vectors as the cosine of the angle between two vectors. It determines whether two vectors are pointing in roughly the same direction. So if the angle between the vectors is 0 degrees, then the cosine similarity is 1.</li>
  <li>Document Vector: The traditional approach to compute text similarity between documents is to do so by transforming the input documents into real-valued vectors. The goal is to have a vector space where similar documents are “close”, according to a chosen similarity measure. This approach takes the name of Vector Space Model, and it’s very convenient because it allows us to use simple linear algebra to compute similarities. We just have to define two things: 
  <ol type='i'>
    <li>A way of transforming documents into vectors</li>
    <li>A similarity measure for vectors</li>
  </ol></li>
  <li>Document Centroid Vector: The simplest way to compute the similarity between two documents using word embeddings is to compute the document centroid vector. This is the vector that’s the average of all the word vectors in the document. Since word embeddings have a fixed size, we’ll end up with a final centroid vector of the same size for each document which we can then use with cosine similarity.</li>
  <li>Word Embeddings: The easiest way of estimating the semantic similarity between a pair of sentences is by taking the average of the word embeddings of all words in the two sentences, and calculating the cosine between the resulting embeddings. </li>
  <li>Sentence Transformers: Sentence-BERT (SBERT) is a modified BERT network that uses siamese and triplet network structures to derive semantically meaningful sentence embeddings. This reduces the effort for finding the most similar pair from 65 hours with BERT / RoBERTa to about 5 seconds with SBERT, while maintaining the accuracy from BERT. </li>
</ul>

<p>I used the technique of Sentence Transformers due to the fact that it not only produces state-of-the-art performance in computing semantic similarity between any two text documents but also takes lesser training times in comparison to standard BERT models which are computationally very expensive.</p>

Installation of Scikit-learn for Task 1: House Price Prediction
------------

scikit-learn requires:

- Python (>= |PythonMinVersion|)
- NumPy (>= |NumPyMinVersion|)
- SciPy (>= |SciPyMinVersion|)
- joblib (>= |JoblibMinVersion|)
- threadpoolctl (>= |ThreadpoolctlMinVersion|)

**Scikit-learn 0.20 was the last version to support Python 2.7 and Python 3.4.**
scikit-learn 1.0 and later require Python 3.7 or newer.
scikit-learn 1.1 and later require Python 3.8 or newer.

Scikit-learn plotting capabilities (i.e., functions start with ``plot_`` and
classes end with "Display") require Matplotlib (>= |MatplotlibMinVersion|).
For running the examples Matplotlib >= |MatplotlibMinVersion| is required.
A few examples require scikit-image >= |Scikit-ImageMinVersion|, a few examples
require pandas >= |PandasMinVersion|, some examples require seaborn >=
|SeabornMinVersion| and plotly >= |PlotlyMinVersion|.

### User installation

If you already have a working installation of numpy and scipy,
the easiest way to install scikit-learn is using `pip`:

    pip install -U scikit-learn

or `conda`:

    conda install -c conda-forge scikit-learn

<p>The documentation includes more detailed `installation instructions <https://scikit-learn.org/stable/install.html>`_.</p>

## Installation of Sentence Transformers for Task 2: Product Matching

We recommend **Python 3.6** or higher, **[PyTorch 1.6.0](https://pytorch.org/get-started/locally/)** or higher and **[transformers v4.6.0](https://github.com/huggingface/transformers)** or higher. The code does **not** work with Python 2.7.

**Install with pip**

Install the *sentence-transformers* with `pip`:

```
pip install -U sentence-transformers
```

**Install with conda**

You can install the *sentence-transformers* with `conda`:

```
conda install -c conda-forge sentence-transformers
```

**Install from sources**

Alternatively, you can also clone the latest version from the [repository](https://github.com/UKPLab/sentence-transformers) and install it directly from the source code:

````
pip install -e .
```` 

**PyTorch with CUDA**

If you want to use a GPU / CUDA, you must install PyTorch with the matching CUDA Version. Follow
[PyTorch - Get Started](https://pytorch.org/get-started/locally/) for further details how to install PyTorch.

  
 
