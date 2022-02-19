import streamlit as st
from PIL import Image

with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

#####################
# Header 
st.write('''
# Ajay Saini, M.Sc.
##### *Resume* 
''')

image = Image.open('dp.png')
st.image(image, width=150)

st.markdown('## Summary', unsafe_allow_html=True)
st.info('''
- Experienced Educator, Researcher and Administrator with almost twenty years of experience in data-oriented environment and a passion for delivering insights based on predictive modeling. 
- Strong verbal and written communication skills as demonstrated by extensive participation as invited speaker at `10` conferences as well as publishing 149 research articles.
- Strong track record in scholarly research with H-index of `32` and total citation of 3200+.
''')

#####################
# Navigation

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #16A2CB;">
  <a class="navbar-brand" href="https://www.linkedin.com/in/the-ajay-ai/" target="_blank">Ajay Saini</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item active">
        <a class="nav-link disabled" href="/">Home <span class="sr-only">(current)</span></a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#education">Education</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#work-experience">Work Experience</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#bioinformatics-tools">Bioinformatics Tools</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#social-media">Social Media</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)

#####################
# Custom function for printing text
def txt(a, b):
  col1, col2 = st.columns([4,1])
  with col1:
    st.markdown(a)
  with col2:
    st.markdown(b)

def txt2(a, b):
  col1, col2 = st.columns([1,4])
  with col1:
    st.markdown(f'`{a}`')
  with col2:
    st.markdown(b)

def txt3(a, b):
  col1, col2 = st.columns([1,2])
  with col1:
    st.markdown(a)
  with col2:
    st.markdown(b)
  
def txt4(a, b, c):
  col1, col2, col3 = st.columns([1.5,2,2])
  with col1:
    st.markdown(f'`{a}`')
  with col2:
    st.markdown(b)
  with col3:
    st.markdown(c)

#####################
st.markdown('''
## Education
''')

txt('**Master of Science** (Computer Science), *Central University of Rajasthan*, Rajasthan',
'2019-2021')
st.markdown('''
- GPA: `8.0(≈)`
- Graduated with First Class Honors.
''')

txt('**Bachelors of Science** (Computer Science), *University of Delhi*, New Delhi',
'2016-2019')
st.markdown('''
- GPA: `7.0(≈)`
- Graduated with First Class Honors.
''')

#####################
st.markdown('''
## Work Experience
''')
txt('**Associate Data Scientist**, [Celebal Technologies](https://celebaltech.com)',
'FAb/2022-Present(Full-Time)')
# st.markdown('''
  
# ''')

txt('**AI Intern**, [Dataviv Technologies](https://dataviv.in)',
'OCT/2021-JAN/2022(Full-Time)')
st.markdown('''
- `Project:` AI BASED SMART ATTENDANCE SYSTEM FOR HOSPITAL
- `Role & Responsibility:` Focused on developing machine learning models for face-recognition and making attendance into the DB with name, time, temperate, And worked on Big-Data Clustering problem using PySaprk, from setup Yarn Cluster and PySpark manually and along with that deals with Geodata to make cluster using DBSCAN Algorithm.Instance Segmentation on custom dataset.

''')

txt('**Data Science Intern**, [Solytics Partners Private Limited](https://solytics-partners.com)',
'JUN/2021-OCT/2021(Full-Time)')
st.markdown('''
- `Project:` Nimbus
- `Role & Responsibility:` Focused on developing machine learning models, models testing.  I was worked on Model Testing and Model Interpretation module of this project.
- `Big Data ML Model:` Worked on Logistic regression, LinerSVC &  Random forest classifier model in PySpark Framework. H2OXGBoostClassifier, H2ODeepLearningClassifier ML Model in H2O.ai Framework.
- `Model Testing:` Use classification model matrices like Confusion Matrix, Precision, Recall, f1-score, Specificity, Sensitivity, Feature Importance.
- `Project:` Sanctions and Adverse Media Screening(SAMS)
- `Sentiment Classifier:`In this project we will be building a sentiment classifier using the bert pretrained model,Machine Learning Algorithm.
- `Spam Classifier:` In this project we will be building a Spam classifier using Machine Learning Algorithm. 
''')

st.markdown('''
## Skills
''')
txt3('Programming', '`Python`, `Linux`')
txt3('Libraries/FrameWork', '`Sklearn`,`Detecton2`,`TensorFlow`,`NLTK`,`Gensim`, `OpenCV`,`Numpy`, `Pandas`, `PySpark`, `Matplotlib`,`Seaborn`')
txt3('Machine Learning', '`Regression Algorithm`, `KNN`, `SVM`, `Decision Tree`, `Random Forests`,etc. ')
txt3('Deep Learning', '`ANN`, `CNN`, `RNN`,`LSTM`')
txt3('Computer Vison', '`OpenCV`,`Detecton2`,')
txt3('NLP', '`Bag-TO-word`, `TF-IDF`, `Word2Vec`, `Word Embedding`')
# txt3('Web development', '`Flask`, `Django`,`HTML`, `CSS`')
txt3('Platforms & Misc.', '`streamlit`, `gradio`, `Anaconda`, `Jupyter Notebook`, `Colab`, `Spyder IDE`, `PyCharm IDE`, `LabelIMG`, `LabelMe`')

#####################
st.markdown('''
## Social Media
''')
txt2('LinkedIn', 'https://www.linkedin.com/in/the-ajay-ai/')
txt2('GitHub', 'https://github.com/the-ajay-ai/')
txt2('Twitter', 'https://twitter.com/ajay_dduc')
txt2('ResearchGate', 'https://www.researchgate.net/profile/Ajay-Saini-7')