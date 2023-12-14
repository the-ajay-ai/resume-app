import streamlit as st
from PIL import Image

with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

#####################
# Header 
st.write('''
# Ajay Saini, `A.I.Enthusiast`
##### *CV* 
''')

image = Image.open('dp.png')
st.image(image, width=150)

st.markdown('## Summary', unsafe_allow_html=True)
st.info('''
- ❖ Energetic, passionate Data Science Enthusiast.I Aim to use my knowledge of Python Programming, OOP Concepts, and hands-on experience in ML, NN, CNN, and Computer Vision.
- ❖ Over 3+ years of experience comprising activities in all phases of the Data Science Project Life Cycle (like analysis, EDA, Feature Engineering, model creation, model testing, and model deployment).   
- ❖ Microsoft Certified:   
    - • Azure Data Scientist Associate   
    - • Azure AI Fundamentals   
    - • Azure Data Fundamentals   
- ❖ Ability to work in a team environment and individual environment.   
- ❖ Hands-on Azure Data-bricks, Azure ML Studio, Azure DataFactory   
- ❖ Hands-on Docker, Git &GitHub, DevOps, Terraform   
- ❖ Hands-on Exp. in Infra as code (IaC) Terraform, ARM templates, Cloud Formation.  
- ❖ Hands-on Azure DevOps CI/CD Process, GIT Repository, check-in, merge, build, and deploy.   
- ❖ Python (NLTK, OpenCV, NumPy, Pandas, Matplotlib, Scikit-learn, LLM, TensorFlow, HuggingFace, etc.)   
- ❖ Knowledge of Cloud ML Services (AzureML, AWS + Sage-maker ) 
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
      <li class="nav-item">
        <a class="nav-link" href="#education">Education</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#work-experience">Work Experience</a>
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
## Work Experience
''')
txt('**Data Science-Analyst(Full-Time)**, [Absolutdata](https://www.absolutdata.com)',
'FAB/2022-Present')
# st.markdown('''

# ''')

txt('**AI Intern(Full-Time)**, [Dataviv Technologies](https://dataviv.in)',
'OCT-JAN/2022')
st.markdown('''
- `Project:` AI BASED SMART ATTENDANCE SYSTEM FOR HOSPITAL
- `Role & Responsibility:` Focused on developing machine learning models for face-recognition and making attendance into the DB with name, time, temperate, And worked on Big-Data Clustering problem using PySaprk, from setup Yarn Cluster and PySpark manually and along with that deals with Geodata to make cluster using DBSCAN Algorithm.Instance Segmentation on custom dataset.

''')

txt('**Data Science Intern(Full-Time)**, [Solytics Partners Private Limited](https://solytics-partners.com)',
'JUN-OCT/2021')
st.markdown('''
- `Project:` NIMBUS
- `Role & Responsibility:` Focused on developing machine learning models, models testing.  I was worked on Model Testing and Model Interpretation module of this project.
- `Big Data ML Model:` Worked on Logistic regression, LinerSVC &  Random forest classifier model in PySpark Framework. H2OXGBoostClassifier, H2ODeepLearningClassifier ML Model in H2O.ai Framework.
- `Model Testing:` Use classification model matrices like Confusion Matrix, Precision, Recall, f1-score, Specificity, Sensitivity, Feature Importance.
- `Project:` Sanctions and Adverse Media Screening(SAMS)
- `Sentiment Classifier:`In this project we will be building a sentiment classifier using the bert pretrained model,Machine Learning Algorithm.
- `Spam Classifier:` In this project we will be building a Spam classifier using Machine Learning Algorithm. 
''')

#####################
st.markdown('''
## Education
''')

txt('**Master of Science** (Computer Science), [*Central University of Rajasthan*, Rajasthan](https://curaj.ac.in)',
'2019-2021')
st.markdown('''
- GPA: `8.0(≈)`
- Graduated with First Class Honors.
''')

txt('**Bachelors of Science** (Computer Science), [*University of Delhi*, New Delhi](http://dducollegedu.ac.in)',
'2016-2019')
st.markdown('''
- GPA: `7.0(≈)`
- Graduated with First Class Honors.
''')

st.markdown('''
## Skills
''')
txt3('Programming', '`Python`, `Linux`')
txt3('Libraries/FrameWork', '`Sklearn`,`Detecton2`,`TensorFlow`,`NLTK`,`Gensim`, `OpenCV`,`Numpy`, `Pandas`, `PySpark`, `Matplotlib`,`Seaborn`')
txt3('Machine Learning', '`Regression Algorithm`, `KNN`, `SVM`, `Decision Tree`, `Random Forests`, `etc.` ')
txt3('Deep Learning', '`ANN`, `CNN`, `RNN`,`LSTM`')
txt3('Computer Vison', '`OpenCV`,`Detecton2`,')
txt3('NLP', '`Bag-TO-word`, `TF-IDF`, `Word2Vec`, `Word Embedding`')
# txt3('Web development', '`Flask`, `Django`,`HTML`, `CSS`')
txt3('Platforms & Misc.', '`Streamlit`, `Gradio`, `Anaconda`, `Jupyter Notebook`, `Colab`, `Spyder IDE`, `PyCharm IDE`, `LabelIMG`, `LabelMe`')

#####################
st.markdown('''
## Social Media
''')
txt3('', '[LinkedIn](https://www.linkedin.com/in/the-ajay-ai/) || [GitHub](https://github.com/the-ajay-ai/) || [Twitter](https://twitter.com/ajay_dduc) || [ResearchGate](https://www.researchgate.net/profile/Ajay-Saini-7)')
